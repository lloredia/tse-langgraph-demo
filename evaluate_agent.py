from langsmith import Client
import json
from agent_graph import qa_app

client = Client()
dataset_name = "qa-eval-demo"
client.create_dataset(dataset_name)

with open("dataset.json") as f:
    samples = json.load(f)

for sample in samples:
    question = sample["question"]
    expected = sample["expected_answer"]
    result = qa_app.invoke({"question": question})
    actual = result["answer"]

    client.log_example(
        inputs={"question": question},
        outputs={"answer": actual},
        dataset_name=dataset_name,
        expected_outputs={"expected": expected}
    )
