#!/usr/bin/env python3
"""
Sketchy Spots Model  -  NFL · College Football · NBA · College Basketball (spreads) · NHL (moneylines)
=======================================================================================================
Finds lines that "make no sense": a favorite laying more than it should, or an underdog getting fewer
points than it should. The market is respecting that team more than its results say, and we take
that sketchy side.

    Our line   = team power ratings built from this season's scores (opponent-adjusted, recent games
                 weighted more, blowouts capped, last season as a starting point) + home edge.
    Gap        = how much more the market likes a team than our line does.
    Sketchy    = gap inside the sport's BAND.  Below it = normal line. Above it = too big - usually an
                 injury or news, so it is listed but NOT bet.
    Bet        = the team the market likes more, at the DraftKings line. Flat 1u.

Plays LOCK once at fixed times (ET): NFL Tue 9am (that week's games) · CFB Sun 9am (next 7 days) ·
NBA / CBB / NHL daily 8am (that day's games). The first-ever run also locks immediately.

THE LEDGER IS FROZEN. A locked bet's line, price and side are written once and never recalculated.
The closing line is recorded until kickoff, then frozen. The result is graded once from the final score
and never changes after that. Nothing is rebuilt from scratch on later runs.

Data: ESPN scoreboard + game summaries (DraftKings lines), free, no key. Standard library only.
Run:  python3 sketchy_model.py      Flags: --offline  --no-open  --now 2026-11-25T08:05
"""
import json, math, os, re, sys, time, urllib.request, webbrowser
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone, date
from statistics import NormalDist

# ================================ CONFIG ====================================
# band: (min gap, max gap) in points for spreads, in win-probability for NHL moneylines
SPORTS = {
    "nfl": dict(label="NFL", path="football/nfl", q="", mkt="spread", lock=("tue", 9, 0), horizon_days=7,
                season="2026-09-08", prev=("2025-09-04", "2026-01-05"), hca=1.7, half_life=56,
                prior_games=4, regress=0.5, cap=21, unknown_pct=None, band=(2.0, 6.0)),
    "cfb": dict(label="College Football", path="football/college-football", q="&groups=80&limit=500", mkt="spread",
                lock=("sun", 9, 0), horizon_days=7, season="2026-08-27", prev=("2025-08-23", "2026-01-21"),
                hca=2.6, half_life=50, prior_games=3, regress=0.5, cap=28, unknown_pct=5, band=(3.5, 10.0)),
    "nba": dict(label="NBA", path="basketball/nba", q="", mkt="spread", lock=("daily", 8, 0), horizon_days=1,
                season="2026-10-20", prev=("2025-10-21", "2026-04-13"), hca=2.4, half_life=30,
                prior_games=10, regress=0.45, cap=25, unknown_pct=None, band=(2.5, 7.0)),
    "cbb": dict(label="College Basketball", path="basketball/mens-college-basketball", q="&groups=50&limit=500",
                mkt="spread", lock=("daily", 8, 0), horizon_days=1, season="2026-11-02",
                prev=("2025-11-03", "2026-04-07"), hca=3.2, half_life=35, prior_games=6, regress=0.55, cap=25,
                unknown_pct=8, band=(3.0, 8.0)),
    "nhl": dict(label="NHL", path="hockey/nhl", q="&limit=100", mkt="ml", lock=("daily", 8, 0), horizon_days=1,
                season="2026-10-06", prev=("2025-10-07", "2026-04-17"), hca=0.25, half_life=40, prior_games=10,
                regress=0.5, cap=4, unknown_pct=None, band=(0.06, 0.15), sd=2.35, max_fav=-250),
}
DEFAULT_ODDS = -110          # spread price when DraftKings' juice is missing
MIN_PRIOR_GAMES = 10         # last-season teams need 10+ games (drops non-D1 / FCS one-offs)
MIN_NEW_TEAM_GAMES = 4       # teams new this season need 4+ games before they're rated or bet
LOCK_GRACE_MIN = 60          # a missed lock-time run can still lock up to 60 min late
FINAL_AFTER_H = 2.0          # check for finals this long after start
WORKERS = 12
# ============================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "sketchy_data.json")
CACHE_DIR = os.path.join(HERE, "cache")
OUT_HTML = os.path.join(HERE, "sketchy_model.html")
TEMPLATE = os.path.join(HERE, "sketchy_template.html")
SB = "https://site.api.espn.com/apis/site/v2/sports/{}/scoreboard?dates={}{}"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/{}/summary?event={}"
N = NormalDist()
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    ET = timezone(timedelta(hours=-5))
DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


# ------------------------------- shared helpers ----------------------------
def http_get(url):
    err = ""
    for i in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sharp-model)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            err = str(e)
            time.sleep(1.0 * (i + 1))
    raise IOError(err)


FETCH = http_get  # swapped out in tests


def pmap(fn, items):
    out = {}
    items = list(items)
    if not items:
        return out
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except Exception:
                out[futs[f]] = None
    return out


def dget(o, *path):
    for p in path:
        if isinstance(o, list):
            o = o[p] if isinstance(p, int) and len(o) > p else None
        elif isinstance(o, dict):
            o = o.get(p)
        else:
            return None
    return o


def nv_home(h_am, a_am):
    """No-vig home win prob from two American prices."""
    dh, da = dec_from_american(h_am), dec_from_american(a_am)
    if not dh or not da:
        return None
    ih, ia = 1 / dh, 1 / da
    return ih / (ih + ia)


def pct_rank(vals, pct):
    v = sorted(vals)
    if not v:
        return 0.0
    return v[min(len(v) - 1, max(0, int(len(v) * pct / 100)))]


def parse_t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(ET)


def to_num(x):
    try:
        s = str(x).strip().lower().lstrip("ou").replace("+", "")
        if s in ("even", "ev", "pk", "pick"):
            return 100.0 if "ev" in s else 0.0
        return float(s)
    except Exception:
        return None


def dec_from_american(a):
    if a is None or a == 0:
        return None
    return 1 + (a / 100 if a > 0 else 100 / abs(a))


def american(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def fmt_american(d):
    a = american(d)
    return f"+{a}" if a > 0 else str(a)


def last_lock(cfg, now):
    day, h, m = cfg["lock"]
    base = now.replace(hour=h, minute=m, second=0, microsecond=0)
    for k in range(8):
        d = base - timedelta(days=k)
        if d <= now and (day == "daily" or DAYS[d.weekday()] == day):
            return d


def next_lock(cfg, now):
    day, h, m = cfg["lock"]
    base = now.replace(hour=h, minute=m, second=0, microsecond=0)
    for k in range(9):
        d = base + timedelta(days=k)
        if d > now and (day == "daily" or DAYS[d.weekday()] == day):
            return d


def lockable(cfg, commence, now):
    if commence <= now:
        return False
    if cfg["lock"][0] == "daily":
        return commence.date() == now.date()
    return commence - now <= timedelta(days=7)


def date_range(a, b):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def ridge(obs, prior, unknown, hca0, lam, half_life=None, asof=None, iters=80, hca_w=1500):
    """obs: (iso_time, home_id, away_id, neutral, margin, weight). Returns ratings, hca."""
    if asof is not None and half_life:
        obs = [(t, h, a, n, m, w * 0.5 ** (max(0, (asof - parse_t(t)).days) / half_life)) for t, h, a, n, m, w in obs]
    teams = {x for o in obs for x in (o[1], o[2])}
    r = {t: prior.get(t, unknown) for t in teams}
    pr = dict(r)
    hca = hca0
    for _ in range(iters):
        num = den = 0.0
        for t, h, a, n, m, w in obs:
            if not n:
                num += w * (m - r[h] + r[a])
                den += w
        hca = (num + hca_w * hca0) / (den + hca_w)
        S, W = defaultdict(float), defaultdict(float)
        for t, h, a, n, m, w in obs:
            e = 0.0 if n else hca
            S[h] += w * (m + r[a] - e); W[h] += w
            S[a] += w * (r[h] + e - m); W[a] += w
        for k in teams:
            r[k] = 0.5 * r[k] + 0.5 * (S[k] + lam * pr[k]) / (W[k] + lam)
    return r, hca


# ------------------------------- ESPN parsing -------------------------------
def team_info(c):
    t = c.get("team") or {}
    return dict(id=str(t.get("id") or c.get("id") or ""), abbr=(t.get("abbreviation") or "").upper(),
                location=t.get("location") or "", name=t.get("displayName") or "", short=t.get("shortDisplayName") or "")


def ml_pair(o, which):
    h = to_num(dget(o, "moneyline", "home", which, "odds"))
    a = to_num(dget(o, "moneyline", "away", which, "odds"))
    if which == "close" and (h is None or a is None):
        h = to_num(dget(o, "homeTeamOdds", "moneyLine")) if h is None else h
        a = to_num(dget(o, "awayTeamOdds", "moneyLine")) if a is None else a
    return [h, a] if h and a else None


def line_num(x):
    s = str(x if x is not None else "").strip().lower()
    if s in ("pk", "pick", "pick'em", "even", "ev"):
        return 0.0
    try:
        return float(s.replace("+", ""))
    except Exception:
        return None


def spread_pair(o, which):
    """[home_line, home_odds, away_odds] e.g. [-3.5, -110, -110]."""
    l = line_num(dget(o, "pointSpread", "home", which, "line"))
    if l is None and which == "close":
        al = line_num(dget(o, "pointSpread", "away", which, "line"))
        l = -al if al is not None else None
    if l is None:
        return None
    ho = to_num(dget(o, "pointSpread", "home", which, "odds")) or DEFAULT_ODDS
    ao = to_num(dget(o, "pointSpread", "away", which, "odds")) or DEFAULT_ODDS
    return [l, ho, ao]


def odds_obj(lst):
    """DraftKings spread + moneyline (open and current) from an ESPN odds / pickcenter list."""
    if not lst:
        return None
    o = next((x for x in lst if "draft" in str(dget(x, "provider", "name") or "").lower()), lst[0])
    so, sc = spread_pair(o, "open"), spread_pair(o, "close")
    if sc is None and o.get("spread") is not None and o.get("details"):
        det = str(o["details"]).split()
        v = line_num(det[-1]) if det else None
        if v is not None:
            sc = [v if det[0].upper() == str(dget(o, "homeTeamOdds", "team", "abbreviation") or "").upper() else -v,
                  DEFAULT_ODDS, DEFAULT_ODDS]
    mo, mc = ml_pair(o, "open"), ml_pair(o, "close")
    out = {}
    if so or sc:
        out["spread"] = dict(open=so or sc, cur=sc or so)
    if mo or mc:
        out["ml"] = dict(open=mo or mc, cur=mc or mo)
    return out or None


def parse_event(ev):
    stype = dget(ev, "season", "type")
    if stype is not None and stype not in (2, 3):
        return None                                   # preseason / all-star
    comp = (ev.get("competitions") or [{}])[0]
    cs = comp.get("competitors") or []
    H = next((c for c in cs if c.get("homeAway") == "home"), None)
    A = next((c for c in cs if c.get("homeAway") == "away"), None)
    if not H or not A:
        return None
    st = (comp.get("status") or ev.get("status") or {}).get("type") or {}
    name = str(st.get("name") or "")
    status = "final" if st.get("completed") else ("off" if ("POSTPONED" in name or "CANCEL" in name)
                                                  else ("live" if st.get("state") == "in" else "pre"))
    return dict(id=str(ev["id"]), commence=parse_t(ev["date"]).isoformat(), home=team_info(H), away=team_info(A),
                neutral=bool(comp.get("neutralSite")), status=status,
                hs=to_num(H.get("score")), as_=to_num(A.get("score")), odds=odds_obj(comp.get("odds")))


# ------------------------------- cache -------------------------------------
def cache_path(sk):
    return os.path.join(CACHE_DIR, f"{sk}.json")


def load_cache(sk):
    c = {}
    if os.path.exists(cache_path(sk)):
        with open(cache_path(sk)) as f:
            c = json.load(f)
    for k, v in (("games", {}), ("done", []), ("lines", {}), ("prior", None), ("bt", None)):
        c.setdefault(k, v)
    return c


def save_cache(sk, c):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path(sk), "w") as f:
        json.dump(c, f, separators=(",", ":"))


def fetch_dates(cfg, dates):
    res = pmap(lambda d: FETCH(SB.format(cfg["path"], d.strftime("%Y%m%d"), cfg["q"])), dates)
    return {d: [g for g in (parse_event(e) for e in data.get("events", [])) if g]
            for d, data in res.items() if data is not None}


def refresh_games(sk, cfg, c, now, notes):
    start = date.fromisoformat(cfg["season"])
    end = now.date() + timedelta(days=cfg["horizon_days"])
    recent = now.date() - timedelta(days=3)
    need = [d for d in date_range(start, end) if d >= recent or d.isoformat() not in c["done"]]
    got = fetch_dates(cfg, need)
    if need and not got:
        notes.append(f"{cfg['label']}: ESPN unavailable this run.")
    done = set(c["done"])
    for d, games in got.items():
        for g in games:
            old = c["games"].get(g["id"])
            if old and not g.get("odds") and old.get("odds"):
                g["odds"] = old["odds"]              # finished games drop odds on ESPN; keep the last ones seen
            c["games"][g["id"]] = g
        if d < now.date() - timedelta(days=1) and all(g["status"] in ("final", "off") for g in games):
            done.add(d.isoformat())
    c["done"] = sorted(done)


def refresh_lines(sk, cfg, c, limit=6000):
    """DraftKings opening/closing lines for finished games (for the backtest), from game summaries."""
    need = [gid for gid, g in c["games"].items() if g["status"] == "final" and gid not in c["lines"]][:limit]
    res = pmap(lambda gid: FETCH(SUMMARY.format(cfg["path"], gid)), need)
    for gid, sm in res.items():
        if sm is not None:
            c["lines"][gid] = odds_obj(sm.get("pickcenter")) or {}


def build_prior(sk, cfg, c, notes):
    if c["prior"]:
        return
    a, b = (date.fromisoformat(x) for x in cfg["prev"])
    got = fetch_dates(cfg, list(date_range(a, b)))
    obs = []
    for games in got.values():
        for g in games:
            if g["status"] == "final" and g["hs"] is not None:
                m = max(-cfg["cap"], min(cfg["cap"], g["hs"] - g["as_"]))
                obs.append((g["commence"], g["home"]["id"], g["away"]["id"], g["neutral"], m, 1.0))
    cnt = Counter(x for o in obs for x in (o[1], o[2]))
    obs = [o for o in obs if cnt[o[1]] >= MIN_PRIOR_GAMES and cnt[o[2]] >= MIN_PRIOR_GAMES]
    if len(obs) < 50:
        notes.append(f"{cfg['label']}: last season's results unavailable; ratings start at average.")
        c["prior"] = {"ratings": {}, "unknown": 0.0, "n": len(obs)}
        return
    r, _ = ridge(obs, {}, 0.0, cfg["hca"], lam=1.0)
    unk = pct_rank(list(r.values()), cfg["unknown_pct"]) * (1 - cfg["regress"]) if cfg["unknown_pct"] else 0.0
    c["prior"] = {"ratings": {t: round(v * (1 - cfg["regress"]), 3) for t, v in r.items()},
                  "unknown": round(unk, 3), "n": len(obs)}


# ------------------------------- ratings / our line -------------------------
def fit_ratings(c, cfg, asof, before=None):
    prior = (c["prior"] or {}).get("ratings", {})
    unk = (c["prior"] or {}).get("unknown", 0.0)
    cnt = Counter(x for g in c["games"].values() if g["status"] == "final" for x in (g["home"]["id"], g["away"]["id"]))
    ok = lambda t: t in prior or cnt[t] >= MIN_NEW_TEAM_GAMES
    obs = []
    for g in c["games"].values():
        if g["status"] != "final" or g["hs"] is None or (before and parse_t(g["commence"]) >= before):
            continue
        if not (ok(g["home"]["id"]) and ok(g["away"]["id"])):
            continue
        m = max(-cfg["cap"], min(cfg["cap"], g["hs"] - g["as_"]))
        obs.append((g["commence"], g["home"]["id"], g["away"]["id"], g["neutral"], m, 1.0))
    r, hca = ridge(obs, prior, unk, cfg["hca"], cfg["prior_games"], cfg["half_life"], asof)
    n = Counter(x for o in obs for x in (o[1], o[2]))
    return dict(r=r, hca=hca, prior=prior, unk=unk, n=n)


def rated(R, tid):
    return tid in R["prior"] or R["n"].get(tid, 0) >= MIN_NEW_TEAM_GAMES


def expected_margin(g, R):
    rh = R["r"].get(g["home"]["id"], R["prior"].get(g["home"]["id"], R["unk"]))
    ra = R["r"].get(g["away"]["id"], R["prior"].get(g["away"]["id"], R["unk"]))
    return rh - ra + (0.0 if g["neutral"] else R["hca"])


def team_name(g, side, sk):
    t = g[side]
    return t["short"] if sk in ("nba", "nfl", "nhl") and t["short"] else (t["location"] or t["name"])


def assess(g, R, cfg, odds, which="cur"):
    """Compare the market with our line. Returns None (no line / unrated) or a dict describing the spot."""
    if not (rated(R, g["home"]["id"]) and rated(R, g["away"]["id"])):
        return dict(status="unrated team", gap=None)
    E = expected_margin(g, R)
    lo, hi = cfg["band"]
    if cfg["mkt"] == "spread":
        sp = (odds or {}).get("spread", {}).get(which)
        if not sp:
            return None
        L, ho, ao = sp
        gap = (-L) - E                          # + : market likes home more than we do
        side = "home" if gap > 0 else "away"
        line = L if side == "home" else -L
        price = ho if side == "home" else ao
        ours = -E if side == "home" else E      # our line for that same team
        out = dict(side=side, line=line, price=price, ours=round(ours, 1), gap=round(abs(gap), 2), E=round(E, 2), mkt_line=L)
    else:
        ml = (odds or {}).get("ml", {}).get(which)
        if not ml:
            return None
        pm = nv_home(*ml)
        pe = N.cdf(E / cfg["sd"])
        gap = pm - pe
        side = "home" if gap > 0 else "away"
        price = ml[0] if side == "home" else ml[1]
        out = dict(side=side, line=None, price=price, ours=round(pe if side == "home" else 1 - pe, 4),
                   mkt_p=round(pm if side == "home" else 1 - pm, 4), gap=round(abs(gap), 4), E=round(E, 2))
    if out["gap"] < lo:
        out["status"] = "normal line"
    elif out["gap"] > hi:
        out["status"] = "too far off - likely injury/news"
    elif cfg.get("max_fav") and out["price"] < 0 and out["price"] < cfg["max_fav"]:
        out["status"] = f"favorite shorter than {cfg['max_fav']}"
    else:
        out["status"] = "SKETCHY"
    return out


def grade_side(side, line, hs, as_, mkt):
    mine = (hs - as_) if side == "home" else (as_ - hs)
    r = mine + (line or 0.0) if mkt == "spread" else mine
    return "W" if r > 0 else ("L" if r < 0 else "P")


# ------------------------------- backtest -----------------------------------
def backtest(sk, cfg, c, now):
    """Every finished game this season: ratings from earlier days only, DraftKings OPENING line,
    graded at that line. Results grouped by how big the gap was."""
    lo, hi = cfg["band"]
    unit = 1 if cfg["mkt"] == "spread" else 100
    edges = [0, lo, (lo + hi) / 2, hi, 99] if cfg["mkt"] == "spread" else [0, lo, (lo + hi) / 2, hi, 1]
    names = ["Normal (below band)", "Sketchy - lower half", "Sketchy - upper half", "Too far off (not bet)"]
    buckets = [dict(name=n, lo=edges[i], hi=edges[i + 1], n=0, w=0, l=0, p=0, units=0.0) for i, n in enumerate(names)]
    days = defaultdict(list)
    for g in c["games"].values():
        if g["status"] == "final" and g["hs"] is not None and (c["lines"].get(g["id"]) or {}):
            days[parse_t(g["commence"]).date()].append(g)
    tested = 0
    for d in sorted(days):
        cut = datetime(d.year, d.month, d.day, tzinfo=ET)
        R = fit_ratings(c, cfg, cut, before=cut)
        for g in days[d]:
            a = assess(g, R, cfg, c["lines"][g["id"]], which="open")
            if not a or a.get("gap") is None:
                continue
            if cfg["mkt"] == "ml" and cfg.get("max_fav") and a["price"] < cfg["max_fav"]:
                continue
            tested += 1
            res = grade_side(a["side"], a["line"], g["hs"], g["as_"], cfg["mkt"])
            dd = dec_from_american(a["price"]) or 1.909
            pl = (dd - 1) if res == "W" else (-1.0 if res == "L" else 0.0)
            b = next(x for x in buckets if x["lo"] <= a["gap"] < x["hi"]) if a["gap"] < edges[-1] else buckets[-1]
            b["n"] += 1; b[res.lower()] += 1; b["units"] += pl
    for b in buckets:
        b["units"] = round(b["units"], 2)
        b["roi"] = round(b["units"] / (b["w"] + b["l"]), 4) if (b["w"] + b["l"]) else None
        if cfg["mkt"] == "ml":
            b["lo"], b["hi"] = round(b["lo"] * 100, 1), round(min(b["hi"], 1) * 100, 1)
    bet = [b for b in buckets[1:3]]
    W, L, P = (sum(b[k] for b in bet) for k in ("w", "l", "p"))
    U = sum(b["units"] for b in bet)
    return dict(date=now.date().isoformat(), games=tested, buckets=buckets, w=W, l=L, p=P, units=round(U, 2),
                roi=round(U / (W + L), 4) if (W + L) else None)


# ------------------------------- the ledger ---------------------------------
def load_store():
    st = {}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            st = json.load(f)
    for k, v in (("bets", {}), ("closes", {}), ("locks", {}), ("boards", {}), ("ratings", {})):
        st.setdefault(k, v)
    st["notes"] = []
    return st


def save_store(st):
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({k: st[k] for k in ("bets", "closes", "locks", "boards", "ratings")}, f, indent=0)
    os.replace(tmp, DATA_FILE)


def lock_due(st, sk, cfg, now):
    """(due, lock_key). First-ever run of a sport locks immediately; afterwards only at its lock time."""
    if sk not in st["locks"]:
        return True, f"first-run {now.isoformat()}"
    lk = last_lock(cfg, now)
    if lk and st["locks"][sk] != lk.isoformat() and timedelta(0) <= now - lk <= timedelta(minutes=LOCK_GRACE_MIN):
        return True, lk.isoformat()
    return False, None


def update(st, now):
    for sk, cfg in SPORTS.items():
        c = load_cache(sk)
        due, key = lock_due(st, sk, cfg, now)
        open_bets = [b for b in st["bets"].values() if b["sport"] == sk and b["result"] is None]
        if not due and not open_bets:
            continue
        refresh_games(sk, cfg, c, now, st["notes"])
        if due:
            refresh_lines(sk, cfg, c)
            build_prior(sk, cfg, c, st["notes"])
            if (c.get("bt") or {}).get("date") != now.date().isoformat():
                c["bt"] = backtest(sk, cfg, c, now)
            R = fit_ratings(c, cfg, now)
            todays = [g for g in c["games"].values() if g["status"] == "pre" and lockable(cfg, parse_t(g["commence"]), now)]
            board, priced = [], 0
            for g in sorted(todays, key=lambda g: g["commence"]):
                a = assess(g, R, cfg, g.get("odds"))
                row = dict(id=g["id"], game=f"{team_name(g,'away',sk)} @ {team_name(g,'home',sk)}", commence=g["commence"],
                           status="no DraftKings line" if a is None else a["status"])
                if a and a.get("gap") is not None:
                    priced += 1
                    row.update(team=team_name(g, a["side"], sk), line=a["line"], price=a["price"], ours=a["ours"],
                               gap=a["gap"], mkt_p=a.get("mkt_p"))
                    bid = f"{g['id']}|{cfg['mkt']}"
                    if a["status"] == "SKETCHY" and bid not in st["bets"]:
                        # written ONCE - never recalculated
                        st["bets"][bid] = dict(
                            id=bid, sport=sk, mkt=cfg["mkt"], event_id=g["id"], commence=g["commence"],
                            game=row["game"], home=team_name(g, "home", sk), away=team_name(g, "away", sk),
                            side=a["side"], team=row["team"], line=a["line"], price=a["price"],
                            dec=round(dec_from_american(a["price"]) or 1.909, 5), ours=a["ours"], gap=a["gap"],
                            mkt_p=a.get("mkt_p"), units=1.0, locked_at=now.isoformat(), lock_key=key,
                            close_line=None, close_price=None, clv=None, closed_at=None,
                            result=None, pl=None, home_score=None, away_score=None, graded_at=None)
                board.append(row)
            st["boards"][sk] = dict(at=now.isoformat(), rows=board)
            names = {}
            for g in c["games"].values():
                for s in ("home", "away"):
                    names[g[s]["id"]] = g[s]["name"] or g[s]["location"]
            st["ratings"][sk] = dict(hca=round(R["hca"], 2), rows=sorted(
                [dict(team=names.get(t, t), r=round(v, 2), n=R["n"].get(t, 0)) for t, v in R["r"].items() if R["n"].get(t, 0) >= 2],
                key=lambda x: -x["r"])[:400])
            if priced or not todays:
                st["locks"][sk] = key if not key.startswith("first-run") else (last_lock(cfg, now) or now).isoformat()
            else:
                st["notes"].append(f"{cfg['label']}: no DraftKings lines at lock time - will retry next run.")
        # closing line: keep updating until start, then freeze
        for b in [b for b in st["bets"].values() if b["sport"] == sk and b["closed_at"] is None]:
            g = c["games"].get(b["event_id"])
            if not g:
                continue
            started = parse_t(b["commence"]) <= now or g["status"] != "pre"
            if not started and g.get("odds"):
                st["closes"][b["id"]] = dict(t=now.isoformat(), odds=g["odds"])
            elif started:
                cl = (st["closes"].get(b["id"]) or {}).get("odds") or {}
                if b["mkt"] == "spread" and cl.get("spread", {}).get("cur"):
                    L, ho, ao = cl["spread"]["cur"]
                    b["close_line"] = L if b["side"] == "home" else -L
                    b["close_price"] = ho if b["side"] == "home" else ao
                    b["clv"] = round(b["line"] - b["close_line"], 2)          # points better than the close
                elif b["mkt"] == "ml" and cl.get("ml", {}).get("cur"):
                    ph = nv_home(*cl["ml"]["cur"])
                    p = ph if b["side"] == "home" else 1 - ph
                    b["close_price"] = cl["ml"]["cur"][0 if b["side"] == "home" else 1]
                    b["clv"] = round(p * b["dec"] - 1, 4)                     # % vs no-vig close
                b["closed_at"] = now.isoformat()                               # frozen from here on
        # grade ONCE
        for b in [b for b in st["bets"].values() if b["sport"] == sk and b["result"] is None]:
            g = c["games"].get(b["event_id"])
            if not g or (now - parse_t(b["commence"])).total_seconds() < FINAL_AFTER_H * 3600:
                continue
            if g["status"] == "final" and g["hs"] is not None:
                b["home_score"], b["away_score"] = g["hs"], g["as_"]
                b["result"] = grade_side(b["side"], b["line"], g["hs"], g["as_"], b["mkt"])
                b["pl"] = round(b["dec"] - 1, 4) if b["result"] == "W" else (-1.0 if b["result"] == "L" else 0.0)
                b["graded_at"] = now.isoformat()
            elif g["status"] == "off":
                b["result"], b["pl"], b["graded_at"] = "P", 0.0, now.isoformat()
        save_cache(sk, c)


# ------------------------------- output -------------------------------------
def build_page(st, now):
    sports = {}
    for sk, cfg in SPORTS.items():
        d, h, m = cfg["lock"]
        lt = datetime(2000, 1, 1, h, m).strftime("%-I:%M%p").lower()
        lk = last_lock(cfg, now)
        c = load_cache(sk)
        sports[sk] = dict(label=cfg["label"], mkt=cfg["mkt"], band=cfg["band"],
                          lock=("Daily" if d == "daily" else d.capitalize()) + f" {lt} ET",
                          next_lock=next_lock(cfg, now).isoformat(), board=st["boards"].get(sk),
                          bt=c.get("bt"), ratings=st["ratings"].get(sk))
    bets = sorted(st["bets"].values(), key=lambda b: (b["commence"], b["sport"]))
    payload = dict(generated=now.isoformat(), sports=sports, bets=bets, notes=sorted(set(st["notes"]))[:8])
    with open(TEMPLATE) as f:
        html = f.read().replace("/*__DATA__*/null", json.dumps(payload))
    with open(OUT_HTML, "w") as f:
        f.write(html)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    now = datetime.now(ET)
    if "--now" in args:
        now = datetime.fromisoformat(args[args.index("--now") + 1]).replace(tzinfo=ET)
    st = load_store()
    if "--offline" not in args:
        update(st, now)
        save_store(st)
    build_page(st, now)
    bets = list(st["bets"].values())
    g = [b for b in bets if b["result"]]
    print(f"{now:%a %b %-d %-I:%M %p} ET | {len(bets)} bets | {sum(b['pl'] for b in g):+.2f}u on {len(g)} graded")
    for n in st["notes"]:
        print("  note:", n)
    if "--no-open" not in args:
        webbrowser.open("file://" + OUT_HTML)


if __name__ == "__main__":
    main()
