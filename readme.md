# Pokedex using CNN (Image Recognition) and RAG

## Features

- Upload an image of a Pokémon to classify it using the CNN model.
- Automatically embed documents related to the Pokémon for detailed information.
- Generate and display a description of the classified Pokémon.
- Provide an input box for users to ask further questions about the Pokémon.

## Working

- **Pokemon Recognition** : The pokemons are recognized using a CNN (Convolution Neural Network) model which has been trained on a Pokemon Image Dataset of 10 classes (10 Pokemons), the model has been trained on Kaggle using P100 GPU and the notebook can be accessed through the Model Training Folder.

- **RAG** : The model generates descriptions and replies to question and answers by implementing RAG (Retrieval Augmented Generation) on a pdf file which contains information of the 10 pokemons "pokemon_info.pdf"

- **LLM Model** : The LLM model used is Llama-3 8b model and is being inferenced using Groq API for faster working.

- **DB** - For the RAG vector db has been used to create document embeddings.

## Video Demonstration

Watch the video demonstration of the project:

[![Watch the video](https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg)](https://youtu.be/ddFb79D8c7g)
![Image](https://github.com/Wriath18/PokeDex_Streamlit_LLM/blob/main/Images/Screenshot%202024-06-17%20135642.png)
