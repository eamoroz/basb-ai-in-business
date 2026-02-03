import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

let reviews = [];
let apiToken = "";
let sentimentPipeline = null;

const LOG_ENDPOINT = "https://script.google.com/macros/s/AKfycbzvtl4PCYjL4dAb-CMivEmZ_4qjBP1P8l0siybU71lsqEi7wQ5GFljyMMX-0Tn8aJpDow/exec";

const analyzeBtn = document.getElementById("analyze-btn");
const reviewText = document.getElementById("review-text");
const sentimentResult = document.getElementById("sentiment-result");
const loadingElement = document.querySelector(".loading");
const errorElement = document.getElementById("error-message");
const apiTokenInput = document.getElementById("api-token");
const statusElement = document.getElementById("status");

document.addEventListener("DOMContentLoaded", function () {
  loadReviews();

  analyzeBtn.addEventListener("click", analyzeRandomReview);
  apiTokenInput.addEventListener("change", saveApiToken);

  const savedToken = localStorage.getItem("hfApiToken");
  if (savedToken) {
    apiTokenInput.value = savedToken;
    apiToken = savedToken;
  }

  initSentimentModel();
});

async function initSentimentModel() {
  try {
    sentimentPipeline = await pipeline(
      "text-classification",
      "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
    );
  } catch (error) {
    showError("Failed to load sentiment model");
  }
}

function loadReviews() {
  fetch("reviews_test.tsv")
    .then((response) => response.text())
    .then((tsvData) => {
      Papa.parse(tsvData, {
        header: true,
        delimiter: "\t",
        complete: (results) => {
          reviews = results.data
            .map((row) => row.text)
            .filter((text) => typeof text === "string" && text.trim() !== "");
        },
      });
    })
    .catch(() => showError("Failed to load reviews"));
}

function saveApiToken() {
  apiToken = apiTokenInput.value.trim();
  if (apiToken) {
    localStorage.setItem("hfApiToken", apiToken);
  } else {
    localStorage.removeItem("hfApiToken");
  }
}

function analyzeRandomReview() {
  hideError();

  if (!reviews.length || !sentimentPipeline) {
    showError("Model or reviews not ready");
    return;
  }

  const selectedReview =
    reviews[Math.floor(Math.random() * reviews.length)];

  reviewText.textContent = selectedReview;
  loadingElement.style.display = "block";
  analyzeBtn.disabled = true;
  sentimentResult.innerHTML = "";
  sentimentResult.className = "sentiment-result";

  analyzeSentiment(selectedReview)
    .then((result) => displaySentiment(result))
    .finally(() => {
      loadingElement.style.display = "none";
      analyzeBtn.disabled = false;
    });
}

async function analyzeSentiment(text) {
  const output = await sentimentPipeline(text);
  return [output];
}

function displaySentiment(result) {
  let sentiment = "neutral";
  let score = 0.5;
  let label = "NEUTRAL";

  if (Array.isArray(result) && result[0]?.[0]) {
    label = result[0][0].label.toUpperCase();
    score = result[0][0].score;

    if (label === "POSITIVE") sentiment = "positive";
    if (label === "NEGATIVE") sentiment = "negative";
  }

  sentimentResult.classList.add(sentiment);
  sentimentResult.innerHTML = `
    <i class="fas ${getSentimentIcon(sentiment)} icon"></i>
    <span>${label} (${(score * 100).toFixed(1)}% confidence)</span>
  `;

  logEvent({
    review: reviewText.textContent,
    label,
    score,
  });
}

function logEvent({ review, label, score }) {
  const payload = {
    ts_iso: new Date().toISOString(),
    review: review,
    sentiment: `${label} (${(score * 100).toFixed(1)}%)`,
    meta: {
      url: window.location.href,
      userAgent: navigator.userAgent,
      language: navigator.language,
      platform: navigator.platform,
    },
  };

  fetch(LOG_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).catch(() => {});
}

function getSentimentIcon(sentiment) {
  if (sentiment === "positive") return "fa-thumbs-up";
  if (sentiment === "negative") return "fa-thumbs-down";
  return "fa-question-circle";
}

function showError(message) {
  errorElement.textContent = message;
  errorElement.style.display = "block";
}

function hideError() {
  errorElement.style.display = "none";
}
