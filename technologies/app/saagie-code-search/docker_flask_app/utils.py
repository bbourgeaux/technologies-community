import torch
import torch.nn as nn
import csv
import ast
from transformers import AutoTokenizer, AutoModel
import requests
from urllib.parse import urlparse
import os

ACCESS_TOKEN = os.environ['COPILOT_GITHUB_API_KEY']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATABASES_DIR = './data'
TOKENIZER = AutoTokenizer.from_pretrained("distilroberta-base")
PRETRAINED_MODEL_NAME = 'distilroberta-base'
FINETUNED_MODEL_NAME = 'distilroberta_2.86.pt'
MODEL_PATH = './model/' + FINETUNED_MODEL_NAME
TAU = 20.
SEQ_LEN = 128

class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, tokenizer, tau, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer
        self.tau = nn.Parameter(torch.tensor(tau), requires_grad=True) # Trainable parameter tau

    def forward(self, input_ids, attention_mask):
        model_output = self.model(input_ids, attention_mask)
        embeddings = self.mean_pooling(model_output, attention_mask)
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
def get_databases(databases_dir=DATABASES_DIR):
    csv_files = []
    for file in os.listdir(databases_dir):
        if file.endswith(".csv"):
            csv_files.append(os.path.splitext(file)[0].replace('_','/'))
    return csv_files


def load_model():
    """Loads a trained model stored in a .pt file"""
    global model
    # Check the available CUDA devices
    cuda_device_count = torch.cuda.device_count()
    # print("CUDA device count:", cuda_device_count)

    # Load the state_dict and map tensors to the existing CUDA device
    if cuda_device_count > 0:
        map_location = f'cuda:{torch.cuda.current_device()}'
    else:
        map_location = 'cpu'

    state_dict = torch.load(MODEL_PATH, map_location=map_location)

    # Create a new instance of your model
    model = AutoModelForSentenceEmbedding(PRETRAINED_MODEL_NAME,
                                      tokenizer=TOKENIZER,
                                      tau=TAU,
                                      normalize=True).to(device)

    # Load the state_dict to the model
    model.load_state_dict(state_dict)


def load_codes_and_embeddings(databases):
    """Loads the codes and embeddings of all databases"""

    data_dir = 'data/'
    global loaded_codes
    global loaded_paths
    global loaded_embeddings

    loaded_codes = []
    loaded_paths = []
    loaded_embeddings = []

    for database in databases:
        current_codes = []
        current_paths = []
        current_embeddings = []
        csv_file = data_dir + database.replace('/','_') + '.csv'

        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                current_codes.append(row['code'])
                current_embeddings.append(torch.tensor(eval(row['embedding'])))
                if 'github' in database:
                    current_paths.append(row['path'])
                else:
                    current_paths.append(' ')
        loaded_codes.append(current_codes)
        loaded_paths.append(current_paths)
        loaded_embeddings.append(current_embeddings)


def encode_query(query, model, tokenizer, seq_len):
    """Returns the embedding of a given query, encoded by a given encoder, tokenizer, seq_len"""
    query = tokenizer(query, max_length=seq_len, truncation=True, padding=True, return_tensors='pt')
    encoded_query = model(query['input_ids'].to(device), query['attention_mask'].to(device))
    return encoded_query

def get_closest_indexes(query, codes, N):
    """Returns the indices of the N most similar codes to given query in a given list of codes, and their similarity to the query."""
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-10)
    # Compute the cosine similarity of the query and each code
    similarities = [cos_sim(query.float().to(device), code.float().to(device)).tolist() for code in codes]
    # Get the N highest similarities
    N_similarities = sorted(similarities, reverse=True)[:N]
    # Scale the similarities
    N_similarities = scale_similarities(N_similarities)
    
    # Get the indices of the N highest similarities
    N_indices = [i for i, _ in sorted(enumerate(similarities), key=lambda x:x[1], reverse=True)[:N]]
    return N_indices, N_similarities

def scale_similarities(similarities):
    similarities = [round(sim[0]*4/3*100) for sim in similarities]
    def truncate(sim):
        if sim > 99:
            sim = 96
        return sim
    return list(map(truncate,similarities))


def get_results(query, database, databases, N=5):
    """Given a NL query and a database of functions, returns the N most relevant functions and their similarities to the given query in the given database"""
    if database in databases:
        db_index = databases.index(database)
        embeddings, codes, paths = loaded_embeddings[db_index], loaded_codes[db_index], loaded_paths[db_index]
    else:
        return ['Impossible to read the database'],[1.0]
    
    encoded_query = encode_query(query, model, TOKENIZER, SEQ_LEN)
    # Return the N closest codes w/ their similarities
    indices, similarities = get_closest_indexes(encoded_query, embeddings, N)
    colors = get_colors(similarities)

    results = [codes[idx] for idx in indices]
    path_results = [paths[idx] for idx in indices]

    results = [result.replace('\\n', '<br>') for result in results]

    return results, path_results, similarities, colors


def get_colors(similarities):
    """Returns the associated colors (in a HTML format) given a list of similarities."""
    def get_color(sim):
        if sim < 60:
            color = '#f24040'
        elif sim < 75:
            color = '#ffa71a'
        else:
            color = '#41d796'
        return color
    return [get_color(sim) for sim in similarities]


def get_python_files_from_repo(repo_url, access_token):
    # Construct the API endpoint for the repository's contents
    contents_url = repo_url.replace("github.com", "api.github.com/repos") + "/contents"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Define a function to recursively traverse the contents of a directory
    def traverse_directory(directory_url):
        response = requests.get(directory_url, headers=headers)
        response.raise_for_status()
        contents = response.json()

        python_files = []
        for content in contents:
            if content["type"] == "file" and content["name"].endswith(".py"):
                python_files.append(content["download_url"])
            elif content["type"] == "dir":
                python_files.extend(traverse_directory(content["url"]))

        return python_files

    # Recursively traverse the contents of the repository
    python_files = traverse_directory(contents_url)

    return python_files

def get_functions_from_file(file_url):
    # Download the file contents from the URL
    response = requests.get(file_url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download file from URL: {file_url}")
    file_contents = response.text

    # Parse the file contents as an AST
    try:
        parsed = ast.parse(file_contents)
    except SyntaxError:
        raise ValueError(f"Failed to parse file as Python code: {file_url}")
    
    # Traverse the AST to extract all functions
    functions = []
    function_start_lines = []
    def traverse(node, from_class_string):
        if isinstance(node, ast.FunctionDef):
            # Get function text
            function_text = ast.unparse(node).strip() 
            # Get line number of the function definition
            lineno = node.lineno
            if from_class_string is not None:
                # Insert "from class ..."
                function_text = function_text.split("\n")
                function_text.insert(1, f"    # {from_class_string}")
                function_text = "\n".join(function_text)
            # Add updated function text to functions
            functions.append(function_text)
            # Add start line number to function_start_lines
            function_start_lines.append(lineno)

        if isinstance(node, ast.ClassDef):
            # Add a comment to the function specifying the current class
            from_class_string = f"from class {node.name}"
            
        for child in ast.iter_child_nodes(node):
            traverse(child, from_class_string)

    traverse(parsed, None)

    return functions, function_start_lines

def get_functions_from_repo(repo_url):
    all_functions = []
    all_paths = []
    python_files = get_python_files_from_repo(repo_url, ACCESS_TOKEN)

    def get_file_path(repo_url, file_url):
        repo_path = urlparse(repo_url).path.rstrip('/')
        file_path = urlparse(file_url).path
        return file_path.replace(repo_path, '', 1)

    for file in python_files:
        functions, function_start_lines = get_functions_from_file(file)
        for function, function_start_line in zip(functions,function_start_lines):
            all_functions += [function]
            all_paths += [get_file_path(repo_url,file) + '#L' + str(function_start_line)]

    return all_functions, all_paths

def encode_new_database(database_name, functions, paths, tokenizer=TOKENIZER, seq_len=SEQ_LEN):
    data = []
    # Encode all functions
    for function, path in zip(functions, paths):      
        tokenized_code = tokenizer(function, max_length=seq_len, truncation=True, padding=True, return_tensors='pt')
        code = tokenized_code['input_ids']
        code_am = tokenized_code['attention_mask']
        code_out = model(code.to(device),code_am.to(device))

        data.append({'code': function, 'path': path, 'embedding': code_out.tolist()})
        
    # Create file for storing all codes and embeddings
    repo_name = './data/' + database_name.replace('/','_') + '.csv'
    # Write in that file
    with open(repo_name, mode='w', newline='') as file:
        fieldnames = ['code', 'path', 'embedding']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for d in data:
            writer.writerow({'code': d['code'], 'path': d['path'], 'embedding': str(d['embedding'])})
