# 🎙️ Podcast Engine

**Microservice Python pour convertir du texte en podcasts professionnels avec Kokoro TTS**

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## 📋 Vue d'ensemble

Podcast Engine est un microservice FastAPI qui transforme n'importe quel texte (articles, PDFs, livres) en podcasts/audiobooks de qualité professionnelle grâce à :

- **✨ Smart Chunking** : Découpage intelligent du texte (respect des phrases)
- **🎤 Kokoro TTS** : Synthèse vocale haute qualité (9 voix disponibles)
- **⚡ Traitement parallèle** : Génération audio asynchrone
- **🔊 ffmpeg** : Merge professionnel des chunks + metadata
- **🎨 GUI Web** : Interface élégante (Tailwind CSS + Alpine.js)
- **📊 Flexibilité maximale** : 29 paramètres configurables

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│   n8n / API Clients / GUI Web       │
└──────────────┬──────────────────────┘
               │ HTTP POST
               │ {text, metadata, options}
┌──────────────▼──────────────────────┐
│      Podcast Engine (FastAPI)       │
│                                     │
│  1. Text Preprocessing & Chunking  │
│  2. Parallel TTS (Kokoro)          │
│  3. ffmpeg Audio Merge             │
│  4. Metadata Embedding (mutagen)   │
│  5. Return M4B/MP3                 │
└─────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1 : Docker (Recommandé)

```bash
# Build image
cd /home/gilles/serverlab/projects/podcast-engine
docker build -t podcast-engine:latest .

# Run container
docker run -d \
  --name podcast-engine \
  -p 8000:8000 \
  -v /home/gilles/serverlab/data/shared:/data/shared \
  -e PODCAST_ENGINE_KOKORO_TTS_URL=http://serverlabapps-kokorotts-skwerq:8880/v1/audio/speech \
  podcast-engine:latest

# Vérifier logs
docker logs -f podcast-engine
```

### Option 2 : Développement local

```bash
# Installation dépendances
pip install -r requirements.txt

# Installer ffmpeg (si pas déjà fait)
sudo apt install -y ffmpeg poppler-utils

# Lancer serveur
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Accéder aux interfaces

- **API Documentation** : http://localhost:8000/docs
- **GUI Web** : http://localhost:8000/gui
- **Health Check** : http://localhost:8000/health

---

## 📖 Utilisation

### Via API REST

```bash
curl -X POST http://localhost:8000/api/v1/create-podcast \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Votre texte ici...",
    "metadata": {
      "title": "Mon Podcast",
      "author": "John Doe",
      "language": "fr"
    },
    "tts_options": {
      "voice": "af_bella",
      "speed": 1.0
    },
    "audio_options": {
      "format": "m4b",
      "bitrate": "64k"
    }
  }'
```

### Via GUI Web

1. Ouvrir http://localhost:8000/gui
2. Coller votre texte
3. Configurer les options (voix, vitesse, format...)
4. Cliquer sur "Create Podcast"
5. Télécharger le fichier M4B/MP3

### Via n8n Workflow

```yaml
Node 1: HTTP Request
  - Method: POST
  - URL: http://podcast-engine:8000/api/v1/create-podcast
  - Body:
      text: "={{ $json.article_text }}"
      metadata:
        title: "={{ $json.title }}"
        author: "={{ $json.author }}"
```

---

## ⚙️ Configuration

### Variables d'environnement

Créer un fichier `.env` :

```bash
# Application
PODCAST_ENGINE_APP_NAME="Podcast Engine"
PODCAST_ENGINE_DEBUG=false
PODCAST_ENGINE_LOG_LEVEL=INFO

# API
PODCAST_ENGINE_API_PORT=8000
PODCAST_ENGINE_API_WORKERS=2

# Kokoro TTS
PODCAST_ENGINE_KOKORO_TTS_URL=http://serverlabapps-kokorotts-skwerq:8880/v1/audio/speech
PODCAST_ENGINE_KOKORO_TIMEOUT=120

# Storage
PODCAST_ENGINE_STORAGE_BASE_PATH=/data/shared/podcasts
PODCAST_ENGINE_TEMP_DIR=/data/shared/podcasts/jobs
PODCAST_ENGINE_FINAL_DIR=/data/shared/podcasts/final

# Defaults
PODCAST_ENGINE_DEFAULT_VOICE=af_bella
PODCAST_ENGINE_DEFAULT_SPEED=1.0
PODCAST_ENGINE_DEFAULT_BITRATE=64k

# GUI
PODCAST_ENGINE_ENABLE_GUI=true
```

---

## 🎤 Voix disponibles (Kokoro TTS)

| Voice ID | Name | Gender | Accent | Quality |
|----------|------|--------|--------|---------|
| `af_bella` | Bella | Female | American | High |
| `af_sarah` | Sarah | Female | American | High |
| `bf_emma` | Emma | Female | British | High |
| `am_adam` | Adam | Male | American | High |
| `bm_george` | George | Male | British | High |
| ... | ... | ... | ... | ... |

---

## 🔧 Paramètres API

### Metadata (PodcastMetadata)

```json
{
  "title": "string (required)",
  "author": "string",
  "description": "string",
  "language": "fr|en|es|de|it",
  "genre": "Audiobook|Technology|Education",
  "narrator": "Kokoro TTS",
  "tags": ["tag1", "tag2"],
  "cover_image_url": "https://..."
}
```

### TTS Options (TTSOptions)

```json
{
  "voice": "af_bella",
  "speed": 1.0,  // 0.5 - 2.0
  "chunk_size": 4000,  // 1000 - 10000 chars
  "preserve_sentence": true,
  "add_chapter_markers": true,
  "remove_urls": true,
  "remove_markdown": true,
  "pause_between_chunks": 0.5  // seconds
}
```

### Audio Options (AudioOptions)

```json
{
  "format": "m4b|mp3|opus|aac",
  "bitrate": "64k",  // 32k, 64k, 128k, 192k
  "sample_rate": 24000,  // Hz
  "channels": 1,  // 1=mono, 2=stereo
  "embed_cover": true,
  "add_silence_start": 0.5,
  "add_silence_end": 1.0
}
```

### Processing Options (ProcessingOptions)

```json
{
  "async_mode": false,
  "max_parallel_tts": 5,
  "retry_on_error": true,
  "max_retries": 3,
  "cleanup_on_error": true,
  "return_binary": true,
  "save_to_storage": true
}
```

---

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

**Réponse** :
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "services": {
    "kokoro_tts": {
      "status": "healthy",
      "url": "http://..."
    }
  },
  "system": {
    "storage": {
      "available": true,
      "path": "/data/shared/podcasts"
    }
  }
}
```

### Logs

```bash
# Docker
docker logs -f podcast-engine

# Local
tail -f logs/podcast-engine.log
```

---

## 🧪 Tests

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_chunking.py -v
```

---

## 🐛 Troubleshooting

### Kokoro TTS Unreachable

```bash
# Vérifier que Kokoro est actif
docker ps | grep kokoro

# Tester connexion
curl http://serverlabapps-kokorotts-skwerq:8880/health
```

### ffmpeg Not Found

```bash
# Vérifier installation
docker exec podcast-engine ffmpeg -version

# Réinstaller si nécessaire (dans Dockerfile)
RUN apt-get install -y ffmpeg
```

### Storage Permission Denied

```bash
# Corriger permissions
sudo chmod -R 777 /home/gilles/serverlab/data/shared/podcasts/
```

---

## 📚 Intégration n8n

### Workflow Example: Wallabag → Podcast

```yaml
Node 1: Wallabag - Get Unread Articles
Node 2: Extract Text (HTML → Plain Text)
Node 3: HTTP Request → Podcast Engine
  URL: http://podcast-engine:8000/api/v1/create-podcast
  Method: POST
  Body:
    text: "={{ $json.content }}"
    metadata:
      title: "={{ $json.title }}"
      author: "={{ $json.domain_name }}"
Node 4: Upload to Audiobookshelf
Node 5: Archive Wallabag Article
```

---

## 🗺️ Roadmap

- [ ] Support EPUB extraction
- [ ] Multi-voice dialogue mode
- [ ] Voice cloning (ElevenLabs)
- [ ] Chapter markers avec timestamps
- [ ] Prometheus metrics
- [ ] Job queue (Celery)
- [ ] WebSocket progress updates

---

## 🤝 Contribution

Les contributions sont bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/amazing`)
3. Commit vos changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing`)
5. Ouvrir une Pull Request

---

## 📄 License

MIT License - Voir [LICENSE](LICENSE)

---

## 🙏 Remerciements

- **Kokoro TTS** - Synthèse vocale haute qualité
- **FastAPI** - Framework web moderne
- **ffmpeg** - Traitement audio robuste
- **mutagen** - Metadata audio

---

**Made with ❤️ by ServerLab**
