import torch
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_filename = "data_finalized.pickle"
model_dir = "weights"
model_filename = "model.pth"
result_dir = "results"


def predict(model, data):
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(len(data)):
            try:
                graph_data, meta_data = data[i].graph, data[i].metadata
                meta_data = meta_data_to_vector(meta_data)
                graph_data, meta_data = graph_data.to(device), meta_data.to(device)
                output = model(graph_data, meta_data)
                results.append(output)
                print(f"Predict: {i}/{len(data)}", end="\r")
            except:
                continue
    results = torch.cat(results)
    return results.cpu()


def main():
    model = torch.load(os.path.join(root, model_dir, model_filename))
    data = load_data(data_filename)
    for i in range(len(data)):
        results = predict(model, data[i].gamestates)
        torch.save(results, os.path.join(root, result_dir, f"{i}.pt"))
    

if __name__ == '__main__':
    main()