// index.js
// Simple Express replacement for the provided FastAPI app.py
const express = require('express');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 8000;

// In-memory store for shortened URLs
// { shortId: { url: 'https://...', expiry: timestampOrNull } }
const urlStore = {};

// Read templates (we expect page.html and expired.html in the same directory or templates folder)
const templatesDir = path.join(__dirname);
const pagePath = path.join(templatesDir, 'page.html');
const expiredPath = path.join(templatesDir, 'expired.html');

// Load page template once (we'll do a simple replace for {{ title }})
let pageTemplate = '';
try {
  pageTemplate = fs.readFileSync(pagePath, 'utf8');
} catch (err) {
  console.error(`Failed to read ${pagePath}:`, err.message);
  process.exit(1);
}

let expiredTemplate = '';
try {
  expiredTemplate = fs.readFileSync(expiredPath, 'utf8');
} catch (err) {
  console.error(`Failed to read ${expiredPath}:`, err.message);
  process.exit(1);
}

// Middlewares
app.use(cors({
  origin: true,
  credentials: true
}));
app.use(express.urlencoded({ extended: true })); // to parse application/x-www-form-urlencoded
app.use(express.json());

// Serve static files (expects a "static" directory next to this file)
app.use('/static', express.static(path.join(__dirname, 'static')));
// If you have embedded font directory as in the Python app:
app.use('/fonts', express.static(path.join(__dirname, 'static', 'fonts', 'IBM_Plex')));

// Utility: generate a 6-char alphanumeric id
function generateShortId(length = 6) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let id = '';
  for (let i = 0; i < length; i++) {
    id += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return id;
}

// Home page - serve page.html and replace {{ title }} with 'Linky' (or customize via query)
app.get('/', (req, res) => {
  const title = 'Linky';
  const html = pageTemplate.replace(/{{\s*title\s*}}/g, title);
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
});

// POST /shorten - accepts url and expiry (same form field names as your page.html)
app.post('/shorten', (req, res) => {
  const url = req.body.url;
  const expiry = req.body.expiry;

  if (!url) {
    return res.status(400).json({ error: 'Missing url field' });
  }

  // Create a unique short id
  let shortId = generateShortId();
  while (urlStore[shortId]) {
    shortId = generateShortId();
  }

  // Compute expiry timestamp (ms) or null for never
  let expiryTs = null;
  if (expiry && expiry !== 'never') {
    const seconds = parseInt(expiry, 10);
    if (!isNaN(seconds) && seconds > 0) {
      expiryTs = Date.now() + seconds * 1000;
    }
  }

  urlStore[shortId] = {
    url,
    expiry: expiryTs
  };

 const BASE_URL = process.env.BASE_URL || `${req.protocol}://${req.get('host')}`;
const fullShort = `${BASE_URL}/${shortId}`;
  return res.json({ short_url: fullShort });
});

// Redirect route
app.get('/:shortId', (req, res) => {
  const shortId = req.params.shortId;
  const entry = urlStore[shortId];

  if (!entry) {
    return res.status(404).send('Short URL not found');
  }

  // Check expiry
  if (entry.expiry === null || Date.now() < entry.expiry) {
    return res.redirect(entry.url);
  } else {
    // expired - remove and show expired page
    delete urlStore[shortId];
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    return res.send(expiredTemplate);
  }
});

// Simple API to list stored items (for debugging) - optional, remove in production
app.get('/__debug/urls', (req, res) => {
  res.json(urlStore);
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Linky server listening on http://0.0.0.0:${PORT}`);
});

