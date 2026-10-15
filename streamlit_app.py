import json
import math
import random
from collections import deque
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="AI Interview Preparation System",
    page_icon="🎯",
    layout="wide",
)

DATA_FILE = Path(__file__).with_name("questions.json")


@st.cache_data
def load_questions():
    with DATA_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required = {"id", "topic", "skill", "difficulty", "question", "keywords"}
    if not isinstance(data, list) or any(not required.issubset(q) for q in data):
        raise ValueError("questions.json has an unexpected structure.")
    return data


try:
    QUESTIONS = load_questions()
except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
    st.error(f"Could not load questions.json: {exc}")
    st.stop()


# ------------------------- SEARCH ALGORITHMS -------------------------

def graph_from_questions(qs):
    nodes = [q["id"] for q in qs]
    graph = {n: [] for n in nodes}
    by_topic = {}
    for q in qs:
        by_topic.setdefault(q["topic"], []).append(q)
    for q in qs:
        same = by_topic.get(q["topic"], [])
        lower = [x for x in same if x["difficulty"] < q["difficulty"]]
        if lower:
            nearest = max(lower, key=lambda x: x["difficulty"])
            graph[nearest["id"]].append(q["id"])
    return graph


def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path
        for nxt in graph.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])
    return [start]


def dfs(graph, start, goal):
    stack = [[start]]
    visited = set()
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for nxt in reversed(graph.get(node, [])):
            stack.append(path + [nxt])
    return [start]


def ucs(graph, start, goal, questions_by_id):
    frontier = [(0, [start])]
    visited_cost = {}
    while frontier:
        frontier.sort(key=lambda x: x[0])
        cost, path = frontier.pop(0)
        node = path[-1]
        if node == goal:
            return path
        if node in visited_cost and visited_cost[node] <= cost:
            continue
        visited_cost[node] = cost
        for nxt in graph.get(node, []):
            step = max(1, questions_by_id[nxt]["difficulty"])
            frontier.append((cost + step, path + [nxt]))
    return [start]


def ids(graph, start, goal):
    def depth_limited(node, goal, limit, path):
        if node == goal:
            return path
        if limit == 0:
            return None
        for nxt in graph.get(node, []):
            if nxt not in path:
                result = depth_limited(nxt, goal, limit - 1, path + [nxt])
                if result:
                    return result
        return None
    for depth in range(len(graph) + 1):
        result = depth_limited(start, goal, depth, [start])
        if result:
            return result
    return [start]


def bidirectional_search(graph, start, goal):
    if start == goal:
        return [start]
    reverse = {n: [] for n in graph}
    for u, children in graph.items():
        for v in children:
            reverse.setdefault(v, []).append(u)
    front_a, front_b = {start: [start]}, {goal: [goal]}
    visited_a, visited_b = set(front_a), set(front_b)
    while front_a and front_b:
        next_a = {}
        for node, path in front_a.items():
            for nxt in graph.get(node, []):
                if nxt in visited_a:
                    continue
                new_path = path + [nxt]
                if nxt in visited_b:
                    other = front_b[nxt]
                    return new_path + list(reversed(other[:-1]))
                visited_a.add(nxt)
                next_a[nxt] = new_path
        front_a = next_a
        next_b = {}
        for node, path in front_b.items():
            for nxt in reverse.get(node, []):
                if nxt in visited_b:
                    continue
                new_path = path + [nxt]
                if nxt in visited_a:
                    other = front_a.get(nxt)
                    if other:
                        return other + list(reversed(new_path[:-1]))
                visited_b.add(nxt)
                next_b[nxt] = new_path
        front_b = next_b
    return [start]


def greedy_best_first(candidates, weakness_scores):
    if not candidates:
        return None
    return max(candidates, key=lambda q: weakness_scores.get(q["skill"], 0) / max(1, q["difficulty"]))


def heuristic(q, weakness_scores, target_difficulty):
    weakness = weakness_scores.get(q["skill"], 0)
    difficulty_gap = abs(q["difficulty"] - target_difficulty)
    return (100 - weakness) + difficulty_gap * 10


def a_star_select(candidates, weakness_scores, target_difficulty):
    if not candidates:
        return None
    return min(candidates, key=lambda q: q["difficulty"] + heuristic(q, weakness_scores, target_difficulty))


def hill_climbing(candidates, weakness_scores, target_difficulty):
    if not candidates:
        return None
    def score(q):
        return 100 - heuristic(q, weakness_scores, target_difficulty)
    current = random.choice(candidates)
    improved = True
    while improved:
        improved = False
        neighbours = random.sample(candidates, min(4, len(candidates)))
        for n in neighbours:
            if score(n) > score(current):
                current = n
                improved = True
                break
    return current


def simulated_annealing(candidates, weakness_scores, target_difficulty):
    if not candidates:
        return None
    def score(q):
        return 100 - heuristic(q, weakness_scores, target_difficulty)
    current = random.choice(candidates)
    temperature = 10.0
    for _ in range(25):
        candidate = random.choice(candidates)
        delta = score(candidate) - score(current)
        if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.01)):
            current = candidate
        temperature *= 0.88
    return current


# ------------------------- CSP AND LOGICAL REASONING -------------------------

def csp_filter(questions, topic, max_difficulty, asked_ids, weaknesses):
    result = []
    for q in questions:
        if q["id"] in asked_ids:
            continue
        if topic != "All Topics" and q["topic"] != topic:
            continue
        if q["difficulty"] > max_difficulty:
            continue
        result.append(q)
    if not result:
        # Recovery step: relax the difficulty limit but preserve topic and no-repeat constraints.
        for q in questions:
            if q["id"] not in asked_ids and (topic == "All Topics" or q["topic"] == topic):
                result.append(q)
    return result


def backtracking_pick(candidates, asked_ids, required_topic=None):
    assignment = []
    def backtrack(index):
        if len(assignment) == 1:
            return assignment[0]
        if index >= len(candidates):
            return None
        q = candidates[index]
        if q["id"] in asked_ids or (required_topic and q["topic"] != required_topic):
            return backtrack(index + 1)
        assignment.append(q)
        result = backtrack(index + 1)
        if result:
            return result
        assignment.pop()
        return backtrack(index + 1)
    return backtrack(0)


def propositional_forward_chaining(facts, rules):
    inferred = set(facts)
    changed = True
    while changed:
        changed = False
        for premises, conclusion in rules:
            if set(premises).issubset(inferred) and conclusion not in inferred:
                inferred.add(conclusion)
                changed = True
    return inferred


def resolution_demo():
    clauses = [{"P", "Q"}, {"~Q"}]
    resolvents = []
    for c1 in clauses:
        for c2 in clauses:
            for lit in list(c1):
                opposite = "~" + lit if not lit.startswith("~") else lit[1:]
                if opposite in c2:
                    resolvents.append((c1 - {lit}) | (c2 - {opposite}))
    return resolvents


def unify(pattern, fact):
    p, f = pattern.strip(), fact.strip()
    if p == f:
        return {}
    if p.startswith("?"):
        return {p: f}
    if f.startswith("?"):
        return {f: p}
    return None


def evaluate_answer(answer, keywords):
    text = answer.lower().strip()
    if not text:
        return 0
    hits = sum(1 for k in keywords if k.lower() in text)
    score = int((hits / max(1, len(keywords))) * 100)
    if len(text.split()) >= 25:
        score = min(100, score + 10)
    return score


# ------------------------- STREAMLIT APPLICATION -------------------------

def init_state():
    defaults = {
        "page": "home",
        "name": "",
        "selected_topic": "All Topics",
        "difficulty": 2,
        "num_questions": 10,
        "asked_ids": set(),
        "history": [],
        "weakness": {},
        "current_question": None,
        "selection_method": "A* Search",
        "last_score": None,
        "last_answer": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_interview():
    for key in ("asked_ids", "history", "weakness"):
        st.session_state[key] = set() if key == "asked_ids" else []
        if key == "weakness":
            st.session_state[key] = {}
    st.session_state.current_question = None
    st.session_state.last_score = None
    st.session_state.last_answer = ""
    st.session_state.page = "home"


def select_next_question():
    candidates = csp_filter(
        QUESTIONS,
        st.session_state.selected_topic,
        st.session_state.difficulty,
        st.session_state.asked_ids,
        st.session_state.weakness,
    )
    if not candidates:
        return None

    recent = st.session_state.history[-3:]
    avg = sum(x["score"] for x in recent) / len(recent) if recent else 0
    target_difficulty = st.session_state.difficulty
    if avg >= 80:
        target_difficulty = min(3, st.session_state.difficulty + 1)
    elif avg < 45:
        target_difficulty = max(1, st.session_state.difficulty - 1)

    methods = [
        ("BFS", lambda: bfs(graph_from_questions(candidates), candidates[0]["id"], candidates[-1]["id"])),
        ("DFS", lambda: dfs(graph_from_questions(candidates), candidates[0]["id"], candidates[-1]["id"])),
        ("UCS", lambda: ucs(graph_from_questions(candidates), candidates[0]["id"], candidates[-1]["id"], {q["id"]: q for q in candidates})),
        ("IDS", lambda: ids(graph_from_questions(candidates), candidates[0]["id"], candidates[-1]["id"])),
        ("Bidirectional Search", lambda: bidirectional_search(graph_from_questions(candidates), candidates[0]["id"], candidates[-1]["id"])),
    ]
    step = len(st.session_state.history)
    if step < len(methods):
        st.session_state.selection_method = methods[step][0]
        path = methods[step][1]()
        target_id = path[-1] if path else candidates[0]["id"]
        selected = next((q for q in candidates if q["id"] == target_id), None)
        if selected:
            return selected

    mode = step % 4
    if mode == 0:
        st.session_state.selection_method = "Greedy Best-First"
        selected = greedy_best_first(candidates, st.session_state.weakness)
    elif mode == 1:
        st.session_state.selection_method = "A* Search"
        selected = a_star_select(candidates, st.session_state.weakness, target_difficulty)
    elif mode == 2:
        st.session_state.selection_method = "Hill Climbing"
        selected = hill_climbing(candidates, st.session_state.weakness, target_difficulty)
    else:
        st.session_state.selection_method = "Simulated Annealing"
        selected = simulated_annealing(candidates, st.session_state.weakness, target_difficulty)
    return selected or backtracking_pick(candidates, st.session_state.asked_ids)


def start_interview():
    st.session_state.name = st.session_state.name_input.strip() or "Student"
    st.session_state.selected_topic = st.session_state.topic_input
    st.session_state.difficulty = st.session_state.difficulty_input
    st.session_state.num_questions = st.session_state.count_input
    st.session_state.asked_ids = set()
    st.session_state.history = []
    st.session_state.weakness = {}
    st.session_state.page = "question"
    st.session_state.current_question = select_next_question()


init_state()

st.title("🎯 AI-Based Interview Preparation System")
st.caption("Adaptive technical interview practice using Search, Constraint Satisfaction Problems, and Logical Reasoning.")

with st.sidebar:
    st.header("About this project")
    st.write("Practice AI concepts with adaptive question selection and keyword-based answer evaluation.")
    st.markdown("**Methods demonstrated**")
    st.write("BFS • DFS • UCS • IDS • Bidirectional Search • Greedy Best-First • A* • Hill Climbing • Simulated Annealing • CSP • Backtracking • Forward Chaining • Resolution • Unification")
    st.divider()
    st.caption("Evaluation is rule-based and keyword-based; it is not an LLM or semantic grader.")

if st.session_state.page == "home":
    st.subheader("Configure your interview")
    topics = ["All Topics"] + sorted({q["topic"] for q in QUESTIONS})
    with st.form("setup_form"):
        name = st.text_input("Student name", value=st.session_state.name, key="name_input")
        topic = st.selectbox("Interview topic", topics, index=0, key="topic_input")
        difficulty = st.radio("Maximum difficulty", [1, 2, 3], index=1, format_func=lambda d: {1: "Easy", 2: "Medium", 3: "Hard"}[d], horizontal=True, key="difficulty_input")
        count = st.number_input("Number of questions", min_value=1, max_value=min(30, len(QUESTIONS)), value=min(10, len(QUESTIONS)), step=1, key="count_input")
        submitted = st.form_submit_button("Start interview", type="primary", use_container_width=True)
    if submitted:
        start_interview()
        st.rerun()
    st.info(f"Question bank loaded: {len(QUESTIONS)} questions across {len(topics)-1} topics.")

elif st.session_state.page == "question":
    q = st.session_state.current_question
    if q is None:
        st.session_state.page = "results"
        st.rerun()
    else:
        n = len(st.session_state.history) + 1
        st.progress(min(1.0, len(st.session_state.history) / max(1, st.session_state.num_questions)))
        st.caption(f"Question {n} of {st.session_state.num_questions}  ·  Topic: {q['topic']}  ·  Skill: {q['skill']}  ·  Selected by: {st.session_state.selection_method}")
        st.subheader(q["question"])
        with st.form(f"answer_form_{q['id']}"):
            answer = st.text_area("Your answer", height=220, placeholder="Write your explanation here…", key=f"answer_{q['id']}")
            submitted = st.form_submit_button("Submit answer", type="primary", use_container_width=True)
        if submitted:
            score = evaluate_answer(answer, q["keywords"])
            st.session_state.history.append({
                "id": q["id"],
                "skill": q["skill"],
                "topic": q["topic"],
                "score": score,
            })
            st.session_state.asked_ids.add(q["id"])
            old = st.session_state.weakness.get(q["skill"], 50)
            st.session_state.weakness[q["skill"]] = int((old + score) / 2)
            st.session_state.last_score = score
            st.session_state.last_answer = answer
            st.session_state.page = "feedback"
            st.rerun()

elif st.session_state.page == "feedback":
    score = st.session_state.last_score or 0
    st.subheader("Answer evaluation")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Score", f"{score}%")
    with col2:
        label = "Strong understanding" if score >= 75 else "Partial understanding" if score >= 45 else "Needs more practice"
        st.markdown(f"### {label}")
        st.write("The score is based on the presence of expected keywords and a small bonus for detailed answers.")
    facts = {"Answered", st.session_state.history[-1]["skill"]}
    inferred = propositional_forward_chaining(facts, [
        (["Answered"], "Attempted_Interview_Question"),
        (["Attempted_Interview_Question"], "Knowledge_Profile_Updated"),
    ])
    if score < 60:
        st.warning("Logical inference: weak skill detected → schedule targeted practice.")
    else:
        st.success("Logical inference: prerequisite understanding appears satisfactory.")
    with st.expander("Show reasoning state"):
        st.write("Inferred facts:", sorted(inferred))
        st.write("Keywords expected:", next(q["keywords"] for q in QUESTIONS if q["id"] == st.session_state.history[-1]["id"]))
    if st.button("Next question", type="primary", use_container_width=True):
        if len(st.session_state.history) >= st.session_state.num_questions:
            st.session_state.page = "results"
        else:
            st.session_state.current_question = select_next_question()
            st.session_state.page = "question"
        st.rerun()

elif st.session_state.page == "results":
    history = st.session_state.history
    st.subheader("Interview report")
    st.write(f"Student: **{st.session_state.name}** · Questions answered: **{len(history)}**")
    overall = int(sum(x["score"] for x in history) / max(1, len(history)))
    st.metric("Overall score", f"{overall}%")
    st.progress(overall / 100)
    topic_scores = {}
    for item in history:
        topic_scores.setdefault(item["topic"], []).append(item["score"])
    st.markdown("### Topic performance")
    for topic, scores in topic_scores.items():
        avg = int(sum(scores) / len(scores))
        st.write(f"**{topic}:** {avg}%")
        st.progress(avg / 100)
    weak = sorted(st.session_state.weakness.items(), key=lambda x: x[1])[:5]
    st.markdown("### Recommended practice")
    if weak:
        for skill, score in weak:
            st.write(f"- **{skill}** — current estimated mastery: {score}%")
    else:
        st.write("No skill profile available.")
    facts = {"Knows_BFS", "Knows_DFS", "Knows_UCS"}
    inferred = propositional_forward_chaining(facts, [
        (["Knows_BFS", "Knows_DFS", "Knows_UCS"], "Knows_Uninformed_Search"),
        (["Knows_Uninformed_Search"], "Ready_For_Advanced_Search"),
    ])
    with st.expander("AI reasoning demonstration"):
        st.write("Forward chaining inferred:", ", ".join(sorted(inferred)))
        st.write("FOL unification example: `Knows(?student,BFS)` ↔ `Knows(Akhil,BFS)`")
        st.write("Substitution:", unify("Knows(?student,BFS)", "Knows(Akhil,BFS)"))
        st.write(f"Resolution demo generated {len(resolution_demo())} resolvent(s).")
    if st.button("Start a new interview", type="primary", use_container_width=True):
        reset_interview()
        st.rerun()
