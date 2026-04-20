import ollama
import json
import os
from rich.console import Console
from rich.table import Table
import time

console = Console()

with open("data/questions.json") as f:
    questions = json.load(f)

MODELS = ["llama3.2", "mistral", "phi3"]
all_results = {}

for model in MODELS:
    console.print(f"\n[bold blue]Testing {model}...[/bold blue]")
    results = []
    correct_count = 0

    for item in questions:
        start = time.time()
        res = ollama.chat(model=model, messages=[
            {"role": "user", "content": item["q"]}
        ])
        latency = round((time.time() - start) * 1000)
        answer = res['message']['content']
        correct = item["truth"].lower() in answer.lower()
        correct_count += correct

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["q"],
            "expected": item["truth"],
            "got": answer[:100],
            "correct": correct,
            "latency_ms": latency
        })

        status = "[green]PASS[/green]" if correct else "[red]FAIL[/red]"
        console.print(f"  {status} {item['q'][:55]}")

    accuracy = correct_count / len(questions)
    avg_latency = round(sum(r["latency_ms"] for r in results) / len(results))
    all_results[model] = {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": len(questions),
        "avg_latency_ms": avg_latency,
        "results": results
    }
    console.print(f"[bold]{model}: {accuracy:.0%} accuracy, {avg_latency}ms avg latency[/bold]")

table = Table(title="\n3-Model Leaderboard")
table.add_column("Rank", width=6)
table.add_column("Model", style="cyan")
table.add_column("Accuracy", style="green")
table.add_column("Correct/Total")
table.add_column("Avg Latency")

ranked = sorted(all_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
for i, (model, stats) in enumerate(ranked):
    table.add_row(
        str(i+1),
        model,
        f"{stats['accuracy']:.0%}",
        f"{stats['correct']}/{stats['total']}",
        f"{stats['avg_latency_ms']}ms"
    )

console.print(table)

os.makedirs("results", exist_ok=True)
with open("results/leaderboard.json", "w") as f:
    json.dump({
        "models_tested": MODELS,
        "questions_count": len(questions),
        "leaderboard": {m: {k: v for k, v in s.items() if k != "results"}
                       for m, s in all_results.items()}
    }, f, indent=2)

console.print("\n[green]Leaderboard saved to results/leaderboard.json[/green]")