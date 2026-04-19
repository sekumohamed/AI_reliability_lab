import ollama
from rich import print
from rich.table import Table
from rich.console import Console
import json
import os

console = Console()

QUESTIONS = [
    {"q": "Who invented the telephone?", "truth": "Alexander Graham Bell"},
    {"q": "What year did World War 2 end?", "truth": "1945"},
    {"q": "What is the capital of Australia?", "truth": "Canberra"},
    {"q": "Who wrote the Iliad?", "truth": "Homer"},
    {"q": "What is the boiling point of water in Celsius?", "truth": "100"},
    {"q": "Who painted the Mona Lisa?", "truth": "Leonardo da Vinci"},
    {"q": "What planet is closest to the Sun?", "truth": "Mercury"},
    {"q": "How many bones are in the adult human body?", "truth": "206"},
    {"q": "Who discovered penicillin?", "truth": "Alexander Fleming"},
    {"q": "What is the chemical symbol for gold?", "truth": "Au"},
]

results = []
console.print("\n[bold blue]Running hallucination test on llama3.2...[/bold blue]\n")

for item in QUESTIONS:
    res = ollama.chat(model='llama3.2', messages=[
        {"role": "user", "content": item["q"]}
    ])
    answer = res['message']['content']
    correct = item["truth"].lower() in answer.lower()
    results.append({
        "question": item["q"],
        "expected": item["truth"],
        "got": answer[:100],
        "correct": correct
    })

table = Table(title="Hallucination Test Results")
table.add_column("Question", style="cyan", width=40)
table.add_column("Expected", style="green", width=20)
table.add_column("Pass?", width=8)

for r in results:
    status = "[green]PASS[/green]" if r["correct"] else "[red]FAIL[/red]"
    table.add_row(r["question"], r["expected"], status)

console.print(table)

accuracy = sum(r["correct"] for r in results) / len(results)
console.print(f"\n[bold]Accuracy: {accuracy:.0%}[/bold] ({sum(r['correct'] for r in results)}/{len(results)} correct)\n")

os.makedirs("results", exist_ok=True)
with open("results/baseline.json", "w") as f:
    json.dump({"model": "llama3.2", "accuracy": accuracy, "results": results}, f, indent=2)

console.print("[green]Results saved to results/baseline.json[/green]")