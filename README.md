# PromptOptimizer

> A lightweight playground for crafting, benchmarking, and auto-tuning AI prompts.

PromptOptimizer helps you experiment with different prompt-engineering techniques like token range sweeps, few-shot examples, and more, so you can quickly discover the wording that delivers the best model performance.

This will then get integrated into https://github.com/whiteducksoftware/flock


---

## 🚀 Quick Start

1. Create a prompt template, for example:

   ```text
   <t1:20-30> Please answer the user's question: <input> <t2:20-30> <e1:2>
   ```

2. Run the optimizer:

   ```bash
   python optimize.py --template template.txt --dataset benchmarks.json
   ```

3. Inspect the ranked prompts in `results/best_prompts.md`.


---

## ✨ Key Features
- **Token blocks**  
  ` <t1:20-30> ` means “insert 20–30 random or learned tokens here.”  
- **Example (N-shot) blocks**  
  ` <e1:2> ` pulls two ground-truth examples from your benchmark set.  
- **Input placeholder**  
  ` <input> ` is replaced by the user input or benchmark query.
- **Usage cost estimator**  
  gradient descent over 1000 data points over 20 epochs can be quite extensive if you chose gpt-4.5 or o3 as model.
  The app will roughly estimate the cost so people are not "I used your app and now openai wants 2mil dollars from me"

---

## 📦 Setup

```bash
git clone https://github.com/your-org/prompt-optimizer.git
cd prompt-optimizer
pip install -r requirements.txt
````

---



## 🔍 Template Syntax

| Token          | Meaning                           | Example      |
| -------------- | --------------------------------- | ------------ |
| `<tX:min-max>` | Token block X with a length range | `<t1:20-30>` |
| `<eY:N>`       | Example block Y with N examples   | `<e1:2>`     |
| `<input>`      | User or dataset input             | `<input>`    |

---

## 🎯 Expected Result

A ranked list of candidate prompts that satisfy your template constraints while maximizing the chosen evaluation metric (accuracy, BLEU, etc.).

---

## 🤝 Contributing

Pull requests are welcome, feel free to open issues for features, bugs, or ideas.

---

## 📜 License

MIT

