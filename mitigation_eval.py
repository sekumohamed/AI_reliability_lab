import ollama
import json
import os
import time
import wikipedia
from rich.console import Console
from rich.table import Table

console = Console()

with open("data/questions.json") as f:
    questions = json.load(f)[:20]

MODEL = "llama3.2"

def baseline(q):
    res = ollama.chat(model=MODEL, messages=[{"role":"user","content":q}])
    return res['message']['content']

def chain_of_thought(q):
    prompt = f"Think step by step before answering.\n\nQuestion: {q}\n\nAnswer:"
    res = ollama.chat(model=MODEL, messages=[{"role":"user","content":prompt}])
    return res['message']['content']

def self_consistency(q, runs=5):
    answers = []
    for _ in range(runs):
        res = ollama.chat(model=MODEL, messages=[{"role":"user","content":q}])
        answers.append(res['message']['content'])
    return max(set(answers), key=answers.count)

def rag_grounding(q):
    try:
        topic = q.replace("What is","").replace("Who is","").replace("?","").strip()[:30]
        context = wikipedia.summary(topic, sentences=2)
        prompt = f"Use this context to answer:\n{context}\n\nQuestion: {q}"
    except:
        context = ""
        prompt = q
    res = ollama.chat(model=MODEL, messages=[{"role":"user","content":prompt}])
    return res['message']['content']

techniques = {
    "baseline": baseline,
    "chain_of_thought": chain_of_thought,
    "self_consistency": self_consistency,
    "rag_grounding": rag_grounding,
}

results = {t: {"correct": 0, "total": 0} for t in techniques}

for item in questions:
    console.print(f"\n[cyan]Q: {item['q'][:60]}[/cyan]")
    for name, fn in techniques.items():
        try:
            answer = fn(item["q"])
            correct = item["truth"].lower() in answer.lower()
            results[name]["correct"] += correct
            results[name]["total"] += 1
            status = "[green]PASS[/green]" if correct else "[red]FAIL[/red]"
            console.print(f"  {name:20} {status}")
        except Exception as e:
            results[name]["total"] += 1
            console.print(f"  {name:20} [yellow]ERROR: {e}[/yellow]")

table = Table(title="\nMitigation Technique Results")
table.add_column("Technique", style="cyan")
table.add_column("Accuracy", style="green")
table.add_column("Correct/Total")
table.add_column("vs Baseline")

baseline_acc = results["baseline"]["correct"] / results["baseline"]["total"]
for name, stats in results.items():
    acc = stats["correct"] / stats["total"]
    diff = acc - baseline_acc
    diff_str = f"+{diff:.0%}" if diff > 0 else f"{diff:.0%}"
    color = "green" if diff > 0 else "red" if diff < 0 else "white"
    table.add_row(name, f"{acc:.0%}", f"{stats['correct']}/{stats['total']}", f"[{color}]{diff_str}[/{color}]")

console.print(table)

os.makedirs("results", exist_ok=True)
with open("results/mitigation_results.json", "w") as f:
    json.dump(results, f, indent=2)

console.print("\n[green]Saved to results/mitigation_results.json[/green]")