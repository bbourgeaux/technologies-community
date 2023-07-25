# Saagie-HF-ModelServer-TextCLF


## Description
Saagie-HF-ModelServer-TextCLF: Custom app based on Dash/Flask that deploys the deep learning models from HuggingFace and makes predictions via the GUI or API. 


## How to use
To deploy the app: you need to create the app with port `8080` exposed, `Base path variable:SAAGIE_BASE_PATH`, don't select `Use rewrite url` and set the port access as `PROJECT`. 

Once the app is up, you can open the page of port 8080, enter a model for text classification on Hugging Face in `Model Name` on the left, then enter the corresponding `Label` and click `Deploy`.

When the model is successfully deployed, you can enter the sentences to be predicted in `Text Classification` on the right side, the sentences will be split with line breaks. Then click `Predict` to get the predicted results.

> An example is: 
> 
> Model Name:j-hartmann/emotion-english-distilroberta-base
> 
> Label: anger 🤬 | disgust 🤢 | fear 😨 | joy 😀 | neutral 😐 | sadness 😭 | surprise 😲 



You can also use the app via API:
> By replacing 'app-...' with your app url, the examples for the deployment and prediction are: 
> 
> curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"model_dir":"j-hartmann/emotion-english-distilroberta-base:main", "label":"anger|disgust|fear|joy|neutral|sadness|surprise"}' "http://app-...:8080/deploy"
> 
> curl -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"inputs":["Good Movie, best of the year", "Highly recommended","very bad", "worst movie"]}' "http://app-...:8080/predict"


# Saagie-Code-Search


## Description

Saagie Code Search is a web application built using Flask and Python, designed to help developers efficiently search and retrieve Python functions. It provides a user-friendly interface for searching code snippets in a default codebase or in code repositories hosted on GitHub.

## How to use

To deploy the app: you need to create the app with port `5000` exposed, `Base path variable:SAAGIE_BASE_PATH`, and select `Use rewrite url` and set the port access as `PROJECT`. 

In Saagie, create the project environment variable $COPILOT_GITHUB_API_KEY containing your GitHub API key.

Once the app is up, you can open the page of port 5000.

## Search Functionality

Saagie Code Search enables users to search for Python functions within the available codebases. The search functionality supports various search criteria, including function names and keywords. Users can enter their search queries and obtain relevant code snippets as search results.

## Default Codebase

The application comes preloaded with a default codebase containing 10,000 Python functions. Users can search for code snippets directly within this default codebase. The default codebase serves as a starting point for users to explore and find relevant code examples.

## GitHub Repository Integration

Saagie Code Search offers integration with GitHub repositories. To enable this functionality, you need to provide a GitHub access token. The access token should be saved in the environment variable `$COPILOT_GITHUB_API_KEY` in Saagie. This token allows the application to fetch the necessary information from the specified GitHub repositories. By providing the GitHub repository link, the application retrieves all the Python functions from the repository and adds them to a separate codebase within the application. This allows users to search for code within the specified repository.

## Supported Language

Saagie Code Search currently supports searching for Python functions only. The application is optimized to handle Python code specifically and does not support other programming languages.

## Usage Instructions

- To search within the default codebase, simply enter your search query in the search interface, specifying the function name or keywords related to the code you are looking for.
- To search within a specific GitHub repository, provide the application with the link to the repository. Saagie Code Search will extract all Python functions from the repository and create a separate codebase for searching within that repository.
- Browse through the search results to find the desired code snippets. Each search result will provide the function name, code snippet, and relevant metadata to assist in code exploration.