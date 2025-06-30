// In app/plugin/popup.js
const API_BASE_URL = "http://localhost:8080";
const FLASK_API_URL = "http://localhost:8080/predict";


const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const resultsDiv = document.getElementById('results');
const summaryText = document.getElementById('summary-text');
const sentimentChartCanvas = document.getElementById('sentimentChart');


document.addEventListener('DOMContentLoaded', async () => {
    loadingDiv.style.display = 'block';

    try {
        const config = await fetchConfig();
        const YOUTUBE_API_KEY = config.youtubeApiKey;
        if (!YOUTUBE_API_KEY) {
            throw new Error("YouTube API Key not found in server config.");
        }
        
        const videoId = await getCurrentVideoId();
        if (!videoId) {
            throw new Error("Not a YouTube video page.");
        }

        const comments = await fetchYouTubeComments(videoId, YOUTUBE_API_KEY);
        if (comments.length === 0) {
            throw new Error("Could not fetch any comments for this video.");
        }
        
        const predictions = await getSentimentPredictions(comments);
        
        displayResults(predictions, comments.length);

    } catch (err) {
        showError(err.message);
    } finally {
        loadingDiv.style.display = 'none';
    }
});

async function fetchConfig() {
    const response = await fetch(`${API_BASE_URL}/config`);
    if (!response.ok) {
        throw new Error("Could not fetch configuration from API.");
    }
    return await response.json();
}

async function getCurrentVideoId() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = new URL(tab.url);
    if (url.hostname === "www.youtube.com" && url.pathname === "/watch") {
        return url.searchParams.get("v");
    }
    return null;
}

async function fetchYouTubeComments(videoId, apiKey) {
    const url = `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&key=${apiKey}&maxResults=100&order=relevance`;
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error("Failed to fetch comments from YouTube API.");
    }
    const data = await response.json();
    return data.items.map(item => item.snippet.topLevelComment.snippet.textOriginal);
}

async function getSentimentPredictions(comments) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comments: comments })
    });
    if (!response.ok) {
        throw new Error("Failed to get predictions from the sentiment API.");
    }
    const data = await response.json();
    return data.predictions;
}

function displayResults(predictions, totalComments) {
    resultsDiv.style.display = 'block';
    
    const positiveCount = predictions.filter(p => p === 1).length;
    const neutralCount = predictions.filter(p => p === 0).length;
    const negativeCount = predictions.filter(p => p === -1).length;

    summaryText.innerText = `Analyzed ${totalComments} comments. Positive: ${positiveCount}, Neutral: ${neutralCount}, Negative: ${negativeCount}.`;

    new Chart(sentimentChartCanvas, {
        type: 'pie',
        data: {
            labels: ['Positive', 'Neutral', 'Negative'],
            datasets: [{
                data: [positiveCount, neutralCount, negativeCount],
                backgroundColor: ['#28a745', '#007bff', '#dc3545'],
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'top' },
                title: { display: true, text: 'Sentiment Distribution' }
            }
        }
    });
}

function showError(message) {
    errorDiv.style.display = 'block';
    errorDiv.textContent = `Error: ${message}`;
}
