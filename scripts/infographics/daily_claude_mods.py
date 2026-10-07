import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import Board

OUT = Path(__file__).resolve().parents[2] / "static/img/daily/claude-code-mods"


def inside_vs_outside():
    b = Board(1100, 600, "Where a mod sits compared with skills and connectors", "Skills and connectors work from outside; a mod runs inside Claude Code itself")
    b.group(20, 100, 330, 470, "Skill", "blue")
    b.card(45, 150, 280, 120, "What it does", ["Teaches Claude how", "to do a job", "(instructions, steps)"], "blue", size=13)
    b.card(45, 300, 280, 110, "Where it acts", ["Outside the screen:", "changes what Claude knows"], "grey", size=13)
    b.card(45, 440, 280, 100, "Cannot", ["add a button or a panel"], "red", size=13)
    b.group(385, 100, 330, 470, "Connector", "green")
    b.card(410, 150, 280, 120, "What it does", ["Plugs Claude into", "another app or service", "(email, images, files)"], "green", size=13)
    b.card(410, 300, 280, 110, "Where it acts", ["Outside the screen:", "changes what Claude reaches"], "grey", size=13)
    b.card(410, 440, 280, 100, "Cannot", ["redraw Claude Code's UI"], "red", size=13)
    b.group(750, 100, 330, 470, "Mod", "orange")
    b.card(775, 150, 280, 120, "What it does", ["A small add-on that changes", "how Claude Code looks", "and how it behaves"], "orange", size=13)
    b.card(775, 300, 280, 110, "Where it acts", ["Inside Claude Code:", "listens to its events"], "orange", size=13)
    b.card(775, 440, 280, 100, "Can", ["add buttons, panels, bars,", "even a mini-game"], "green", size=13)
    return b


def event_flow():
    b = Board(1100, 640, "How a mod reacts: event, hook, three possible answers", "Every action Claude Code takes announces itself as an event; the mod's hook gets the first look")
    ev = b.card(30, 120, 230, 120, "Event", ["Claude is about to", "run a command,", "edit a file, draw a row"], "blue", size=13)
    hook = b.card(335, 120, 260, 120, "Mod hook", ["small code that", "listens for that", "event"], "orange", size=13)
    eng = b.card(670, 120, 400, 120, "Rest of Claude Code", ["other plugins, then the", "engine's own behaviour", "(the thing that normally happens)"], "grey", size=13)
    b.arrow(ev.right(), hook.left(), label="signal")
    b.arrow(hook.right(), eng.left(), label="next(e)")
    b.group(30, 300, 1040, 300, "What the hook may do", "purple")
    b.card(55, 350, 320, 135, "1. Watch", ["call next(e) unchanged", "and note what happened", "e.g. count tool calls"], "teal", size=13)
    b.card(390, 350, 320, 135, "2. Change", ["call next({...e, edit})", "the rest sees the edit", "e.g. trim a command"], "yellow", size=13)
    b.card(725, 350, 320, 135, "3. Handle itself", ["return without next", "normal behaviour never runs", "e.g. hide a row, deny a call"], "red", size=13)
    b.text(550, 535, "Clean View, as the video describes it: Claude Code is about to show a tool row,", 13, "grey", "400")
    b.text(550, 558, "the mod steps in and hides it, then checks the step off when it finishes.", 13, "grey", "400")
    return b


def clean_view():
    b = Board(1100, 640, "Mod 1: Clean View", "Same prompt, same work; the mod only changes what is drawn")
    b.group(20, 95, 520, 520, "Normal Claude Code", "grey")
    rows = ["> build me a weather dashboard", "  Read  package.json", "  Bash  npm init -y", "  Write src/App.tsx (212 lines)", "  Write src/api.ts (64 lines)", "  Bash  npm run build", "  Edit  src/App.tsx", "  Bash  npm test ...", "  ... many more rows ..."]
    b.card(45, 145, 470, 280, "", rows, "grey", size=13, align="left")
    b.text(280, 470, "Every tool call is a visible row.", 13, "grey", "700")
    b.text(280, 495, "Useful for debugging; noisy if you only want progress.", 12, "grey")
    b.group(560, 95, 520, 520, "With Clean View on", "green")
    b.card(585, 145, 470, 280, "Plan for: weather dashboard", ["[x] understand the request", "[x] design the layout", "[x] fetch New York forecast", "[~] build the dashboard page", "[ ] test and tidy up", "", "2 of 5 steps done"], "green", size=14, align="left")
    b.card(585, 450, 470, 125, "When finished", ["a short summary of what it did", "no tool rows, no code scrolling by"], "teal", size=13)
    return b


def control_panel():
    b = Board(1100, 540, "Mod 2: Control Panel", "Change the model and effort from the toolbar instead of typing slash commands")
    a = b.group(20, 95, 500, 410, "Without the mod", "grey")
    b.card(45, 150, 450, 90, "Switch model", ["type /model, then pick from a list"], "grey", size=13)
    b.card(45, 270, 450, 90, "Change thinking effort", ["type /effort, then pick a level"], "grey", size=13)
    b.card(45, 390, 450, 90, "Cost", ["two commands every time you change your mind"], "red", size=13)
    b.group(580, 95, 500, 410, "With the mod", "orange")
    b.card(605, 150, 450, 90, "Toolbar button at the bottom right", ["click the mod's name to open it"], "orange", size=13)
    b.card(605, 270, 215, 90, "Model", ["pick in place"], "blue", size=13)
    b.card(840, 270, 215, 90, "Effort", ["pick in place"], "purple", size=13)
    b.card(605, 390, 450, 90, "Status bar updates at once", ["you see the new setting immediately"], "green", size=13)
    return b


def agent_dock():
    b = Board(1100, 700, "Mod 3: Agent Dock", "Choose how many helper agents work on the next prompt, and watch each one")
    b.card(25, 95, 330, 150, "Pick before you prompt", ["type /dock", "choose team size (up to 50)", "choose helper type"], "orange", size=13)
    b.card(25, 275, 330, 125, "Helper type", ["fast and cheap model", "or the same model as the lead", "(the video: Opus 5.5)"], "purple", size=13)
    b.card(25, 430, 330, 135, "Warning built into the mod", ["a big team uses your plan's", "usage much faster; the mod", "warns before going too big"], "red", size=13)
    b.group(385, 95, 690, 585, "Dock for: 50 bakery launch tasks (golden-crumb)", "teal")
    work = [("Order ovens", "working", "green"), ("Menu pricing", "working", "green"), ("Hire staff", "working", "green"), ("Permits", "working", "green"),
            ("Signage", "working", "green"), ("Supplier deals", "working", "green"), ("Website", "working", "green"), ("Social posts", "working", "green"),
            ("Launch party", "working", "green"), ("Tasting menu", "working", "green")]
    x0, y0 = 405, 145
    for i, (name, state, col) in enumerate(work):
        cx = x0 + (i % 2) * 335
        cy = y0 + (i // 2) * 62
        b.card(cx, cy, 320, 52, "", [], col, size=12)
        b.text(cx + 14, cy + 31, name, 13, col, "700", anchor="start")
        b.bar(cx + 160, cy + 20, 140, 0.2 + 0.07 * i, color="green", h=12)
    b.card(405, 465, 320, 52, "waiting: 10 more cards", ["queued for a free slot"], "grey", size=11, dashed=True)
    b.card(740, 465, 320, 52, "waiting: 30 more tasks", ["not shown on screen yet"], "grey", size=11, dashed=True)
    b.text(730, 548, "Video's count: 50 chosen, 20 cards visible, 10 working, rest waiting their turn.", 12, "grey")
    b.card(405, 575, 655, 85, "When every card is done", ["the lead agent gathers the results into one dashboard"], "blue", size=13)
    b.arrow((355, 170), (405, 170))
    return b


def photos():
    b = Board(1100, 620, "Mod 4: Claude Photos (built on the Higgsfield connector)", "A full image and video studio inside Claude Code, with a spending check before anything is made")
    steps = [
        ("1. /create", ["opens the studio panel", "credit balance, top right"], "blue"),
        ("2. Describe", ["one sentence:", "me riding a train"], "teal"),
        ("3. Choose", ["photo or video", "Nano Banana Pro", "16:9, 4K, reference photo"], "purple"),
        ("4. Prompt expanded", ["Claude turns it into a", "long, detailed prompt", "for that model (editable)"], "orange"),
    ]
    cards = []
    for i, (t, l, c) in enumerate(steps):
        cards.append(b.card(25 + i * 270, 110, 250, 150, t, l, c, size=13))
    for a, c in zip(cards, cards[1:]):
        b.arrow(a.right(), c.left())
    gate = b.diamond(180, 400, 280, 130, "Credits needed?\nask the person", "yellow", size=13)
    b.arrow(cards[-1].bottom(), gate.top(), via=[(cards[-1].bottom()[0], 300), (180, 300)])
    gen = b.card(400, 345, 200, 110, "5. Generate", ["only after you", "click yes"], "green", size=13)
    prev = b.card(650, 345, 200, 110, "6. Preview", ["small pixel preview", "inside the terminal"], "pink", size=13)
    save = b.card(895, 345, 190, 110, "7. Real file", ["opens the full photo", "already saved locally"], "teal", size=13)
    b.arrow(gate.right(), gen.left(), label="yes")
    b.arrow(gen.right(), prev.left())
    b.arrow(prev.right(), save.left())
    b.card(25, 510, 1060, 80, "Disclosure in the video", ["Higgsfield sponsored the video, and this mod runs on top of their connector. The mod itself is a pattern, not a requirement of any one provider."], "grey", size=13)
    return b


def roadtrip():
    b = Board(1100, 460, "Mod 5: Roadtrip (a game while Claude builds)", "A mod can add something fun to the waiting time")
    c1 = b.card(30, 110, 300, 130, "Claude is working", ["e.g. improving your", "weather dashboard", "(takes a while)"], "blue", size=13)
    c2 = b.card(400, 110, 300, 130, "You click Play", ["a mini racing game", "opens inside Claude Code"], "orange", size=13)
    c3 = b.card(770, 110, 300, 130, "Quiz break", ["every so often, a question", "about Claude appears"], "purple", size=13)
    b.arrow(c1.right(), c2.left())
    b.arrow(c2.right(), c3.left())
    b.card(30, 290, 1040, 110, "What it shows about mods", ["Mods can draw their own panels, so this one spends the space on a game.", "The pane is only for the wait: Claude keeps building underneath."], "teal", size=13)
    return b


def session_bar():
    b = Board(1100, 420, "Bonus: Session bar (shown, not yet explained)", "Left-hand bar listing live sessions, with one-click jumping")
    b.card(30, 110, 300, 190, "Session bar", ["list of your sessions", "see which are running", "click one to jump in"], "green", size=14, align="left")
    b.card(400, 110, 300, 190, "What the video says", ["see which sessions are running", "jump straight into one", "built by the creator and used daily"], "blue", size=13)
    b.card(770, 110, 300, 190, "Status", ["no build steps given", "the creator offered a", "separate video if people ask"], "yellow", size=13)
    b.arrow((330, 205), (400, 205))
    b.arrow((700, 205), (770, 205))
    b.text(550, 360, "This is why the video says six mods in the intro but numbers five: the session bar is the sixth.", 13, "grey")
    return b


def build_loop():
    b = Board(1100, 640, "Build your own mod: the loop", "You describe it; Claude Code writes it; you reload and iterate")
    s = [
        ("1. Update", ["Claude Code must be a", "recent build with mods", "(video: 2.1.287 or newer)"], "blue"),
        ("2. Describe", ["say the feature you wish", "existed, in plain words"], "teal"),
        ("3. Build", ["Claude Code writes the mod", "using its built-in skill", "for authoring mods"], "purple"),
        ("4. Reload", ["it asks to reload;", "say yes; the mod shows", "up when the turn ends"], "orange"),
    ]
    cs = []
    for i, (t, l, c) in enumerate(s):
        cs.append(b.card(25 + i * 270, 110, 250, 140, t, l, c, size=13))
    for a, c in zip(cs, cs[1:]):
        b.arrow(a.right(), c.left())
    it = b.card(25, 320, 500, 120, "5. Iterate in plain English", ["\"too noisy\", \"move it left\", \"warn me first\"", "Claude edits the mod; it reloads again"], "yellow", size=13)
    b.arrow(cs[3].bottom(), it.top(), via=[(cs[3].bottom()[0], 285), (275, 285)])
    b.group(560, 300, 515, 310, "Keep it?", "green")
    b.card(585, 345, 215, 130, "Default: temporary", ["lives in the session's", "mods folder only"], "grey", size=13)
    b.card(835, 345, 215, 130, "Save as plugin", ["ask Claude to save", "and install it, so it", "loads every session"], "green", size=13)
    b.arrow((800, 410), (835, 410))
    b.card(585, 500, 465, 85, "Sharing", ["a mod is a plugin: others use /plugin to find and install it"], "teal", size=13)
    return b


def mod_anatomy():
    b = Board(1100, 600, "Extra: what a mod is made of", "Not from the video: from Claude Code's own mod-authoring reference")
    b.group(20, 95, 520, 490, "Three files in a folder", "blue")
    b.card(45, 145, 470, 110, ".claude-plugin/plugin.json", ["name, version, description"], "blue", size=13)
    b.card(45, 285, 470, 110, "hooks/hooks.json", ["names one module: ./register.tsx"], "teal", size=13)
    b.card(45, 425, 470, 130, "hooks/register.tsx", ["export const register = (on) => {", "  on(event, matcher, hook)", "}"], "orange", size=13, align="left")
    b.group(570, 95, 510, 490, "The ask decides the API", "purple")
    rows = [["You ask for", "A mod uses"],
            ["hide or reshape rows", "ui.render hook"],
            ["a panel / sidebar", "$.ui.open + Pane"],
            ["a bar above prompt", "AbovePrompt"],
            ["a slash command", "$.command.register"],
            ["block or edit a call", "tool.call hook"],
            ["react to a prompt", "prompt.submit hook"],
            ["do work later", "$.clock timers"]]
    b.table(590, 145, [230, 240], rows, "purple", size=13)
    return b


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("inside-vs-outside", inside_vs_outside), ("event-flow", event_flow), ("clean-view", clean_view),
                     ("control-panel", control_panel), ("agent-dock", agent_dock), ("claude-photos", photos),
                     ("roadtrip", roadtrip), ("session-bar", session_bar), ("build-loop", build_loop), ("mod-anatomy", mod_anatomy)]:
        fn().save(OUT / f"{name}.svg")
