from langsmith import Client
from langsmith.evaluation import StringEvaluator

client = Client()
dataset_name = "qa-eval-demo"
examples = client.list_examples(dataset_name=dataset_name)

evaluator = StringEvaluator(name="exact-match")

for example in examples:
    run = client.read_run(example.run_id)
    result = evaluator.evaluate_run(run)
    print(f"{example.inputs['question']} → Score: {result.score}")
