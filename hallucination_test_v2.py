import ollama
import json
import os
from rich.console import Console
from rich.table import Table

console = Console()

with open("data/questions.json") as f:
    questions = json.load(f)

results = []
categories = {}

console.print("\n[bold blue]Running 50-question hallucination test...[/bold blue]\n")

for item in questions:
    res = ollama.chat(model='llama3.2', messages=[
        {"role": "user", "content": item["q"]}
    ])
    answer = res['message']['content']
    correct = item["truth"].lower() in answer.lower()
    cat = item["category"]

    results.append({
        "id": item["id"],
        "category": cat,
        "question": item["q"],
        "expected": item["truth"],
        "got": answer[:120],
        "correct": correct
    })

    if cat not in categories:
        categories[cat] = {"total": 0, "correct": 0}
    categories[cat]["total"] += 1
    categories[cat]["correct"] += correct

    status = "[green]PASS[/green]" if correct else "[red]FAIL[/red]"
    console.print(f"{status} [{cat}] {item['q'][:60]}")

table = Table(title="\nCategory Breakdown")
table.add_column("Category", style="cyan")
table.add_column("Correct", style="green")
table.add_column("Total")
table.add_column("Accuracy")

for cat, stats in categories.items():
    acc = stats["correct"] / stats["total"]
    table.add_row(cat, str(stats["correct"]), str(stats["total"]), f"{acc:.0%}")

console.print(table)

overall = sum(r["correct"] for r in results) / len(results)
console.print(f"\n[bold]Overall Accuracy: {overall:.0%}[/bold] ({sum(r['correct'] for r in results)}/50)\n")

failures = [r for r in results if not r["correct"]]
console.print(f"[red]Failed questions: {len(failures)}[/red]")
for f in failures:
    console.print(f"  - [{f['category']}] {f['question']} → expected: {f['expected']}")

os.makedirs("results", exist_ok=True)
with open("results/50q_results.json", "w") as f:
    json.dump({"model": "llama3.2", "overall_accuracy": overall,
               "categories": categories, "results": results}, f, indent=2)

console.print("\n[green]Results saved to results/50q_results.json[/green]")