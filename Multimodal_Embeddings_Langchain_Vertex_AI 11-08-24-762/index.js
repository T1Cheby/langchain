import fs from "fs"; // Import the Node.js file system module
import path from "path"; // Import the Node.js path module
import { GoogleVertexAIMultimodalEmbeddings } from "langchain/experimental/multimodal_embeddings/googlevertexai"; // Import Google Vertex AI Multimodal Embeddings
import { FaissStore } from "langchain/vectorstores/faiss"; // Import FaissStore (Faiss used for similarity search ) for vector storage 
import { Document } from "langchain/document"; // Import Document class for creating documents

// Initialize GoogleVertexAIMultimodalEmbeddings
const embeddings = new GoogleVertexAIMultimodalEmbeddings();

// Define path for vector store
const vectorStorePath = "vector_store";

// Function to clear a directory
const clearDirectory = (directoryPath) => { 
  fs.readdirSync(directoryPath).forEach((file) => { // Read contents of the directory synchronously
    const filePath = path.join(directoryPath, file); // Construct full path of the file
    fs.unlinkSync(filePath); // Delete the file
  });
};

// Function to add image and its embeddings to the vector store
const addImage = async (path, id) => {
  const img = fs.readFileSync(path); // Read image file synchronously
  const vectors = await embeddings.embedImageQuery(img); // Embed the image using Google Vertex AI
  const document = new Document({
    pageContent: img.toString("base64"), // Encode image to base64
    metadata: { id: id, mediaType: "image", path: path }, // Metadata for the image document
  });
  await vectorStore.addVectors([vectors], [document]); // Add vectors and document to vector store
  console.log(`Image ${path} added.`); // Log success message
};

// Function to add text and its embeddings to the vector store
const addText = async (text, id) => {
  const vectors = await embeddings.embedQuery(text); // Embed the text using Google Vertex AI
  const document = new Document({
    pageContent: text, // Text content
    metadata: { id: id, mediaType: "text" }, // Metadata for the text document
  });
  await vectorStore.addVectors([vectors], [document]); // Add vectors and document to vector store
  console.log(`Text "${text}" added.`); // Log success message
};

// Function to perform image similarity search
const imageSimilaritySearch = async (path, results) => {
  clearDirectory("output"); // Clear the output directory
  console.log("Performing image similarity search."); // Log message
  const imageQuery = fs.readFileSync(path); // Read image file synchronously
  const imageVectors = await embeddings.embedImageQuery(imageQuery); // Embed the image using Google Vertex AI
  const imageResult = await vectorStore.similaritySearchVectorWithScore(
    imageVectors,
    results //number of simliar results 
  ); // Perform similarity search
  console.log(`Similarity search results for image query: ${path}`); // Log message
  printResults(imageResult); // Print search results
};

// Function to perform text similarity search
const textSimilaritySearch = async (text, results) => {
  clearDirectory("output"); // Clear the output directory
  console.log("Performing text similarity search."); // Log message
  const textVectors = await embeddings.embedQuery(text); // Embed the text using Google Vertex AI
  const textResult = await vectorStore.similaritySearchVectorWithScore(
    textVectors,
    results
  ); // Perform similarity search
  console.log(`Similarity search results for text query: ${text}`); // Log message
  printResults(textResult); // Print search results
};

// Function to print similarity search results
const printResults = (results) => {
  results.forEach((item) => { // Iterate through search results
    const metadata = JSON.stringify(item[0].metadata); // Convert metadata to JSON string
    console.log(`${metadata}`); // Log metadata
    if (item[0].metadata.mediaType === "text") { // If media type is text
      console.log(`Text: ${item[0].pageContent}`); // Log text content
    } else if (item[0].metadata.mediaType === "image") { // If media type is image
      let filename = item[0].metadata.path.split("/").pop(); // Extract filename from path
      fs.writeFileSync(
        `output/${filename}`,
        item[0].pageContent,
        "base64"
      ); // Write image file to output directory
    }
  });
};

// Initialize vector store
let vectorStore = fs.existsSync(vectorStorePath)
  ? await FaissStore.load(vectorStorePath) // Load vector store if it exists
  : new FaissStore(embeddings, {}); // Otherwise create a new vector store

// Function to add items to the vector store
const addItemsToVectorStore = async () => {
  if (!fs.existsSync(vectorStorePath)) { // If vector store does not exist
    const images = [ // Array of image paths
      "images/dog.jpeg",
      "images/cat.jpg",
      "images/parrot.jpg",
      "images/iphone.jpeg",
      "images/steve.jpeg",
      "images/airpod.jpeg",
    ];
    const texts = [ // Array of text content
      "Dogs are domesticated mammals.",
      "Apple Inc. is an American multinational technology company.",
      "Steve Jobs was the chairman, chief executive officer, and co-founder of Apple Inc.",
    ];

    let imageId = 0;
    for (let i = 0; i < images.length; i++, imageId++) // Iterate through images
      await addImage(images[i], imageId); // Add each image to vector store

    let textId = images.length;
    for (let i = 0; i < texts.length; i++, textId++) // Iterate through texts
      await addText(texts[i], textId); // Add each text to vector store
  }
};

// Add items to the vector store
await addItemsToVectorStore();

// Perform similarity searches
// await imageSimilaritySearch('images/dog2.jpeg', 1); // Search for similar images 
// await imageSimilaritySearch('images/steve2.jpg', 1); 
await textSimilaritySearch('Mammals', 1); // Search for similar text
// await textSimilaritySearch('Apple Inc', 1);
// await textSimilaritySearch('Steve Jobs', 1);

// Save the state of the vector store
await vectorStore.save(vectorStorePath); // Save vector store to file system
