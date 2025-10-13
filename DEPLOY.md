# 🚀 Déploiement Podcast Engine dans Dokploy

Guide étape par étape pour déployer le service Podcast Engine dans Dokploy.

---

## 📋 Prérequis

- ✅ Image Docker buildée : `podcast-engine:latest`
- ✅ Dokploy installé et accessible
- ✅ Kokoro TTS déjà déployé dans Swarm (`serverlabapps-kokorotts-skwerq`)
- ✅ Volume `/home/gilles/serverlab/data/shared/podcasts/` créé sur host

---

## 🛠️ Étape 1 : Créer l'application dans Dokploy

### Via GUI Dokploy (Recommandé)

1. **Ouvrir Dokploy** : https://dokploy.robotsinlove.be

2. **Créer nouvelle application** :
   - Aller dans le projet `serverlabapps` (ou créer nouveau projet)
   - Cliquer sur **"+ New Service"**
   - Sélectionner **"Docker"** (pas Docker Compose)

3. **Configuration de base** :
   ```
   Name: podcast-engine
   Description: Text-to-Podcast conversion service with Kokoro TTS
   ```

### Configuration Image

```yaml
Image Source: Local Image
Image Name: podcast-engine:latest
Tag: latest
Pull Policy: IfNotPresent
```

---

## ⚙️ Étape 2 : Variables d'environnement

Ajouter ces variables dans l'onglet **"Environment"** :

```bash
# Application
PODCAST_ENGINE_APP_NAME=Podcast Engine
PODCAST_ENGINE_DEBUG=false
PODCAST_ENGINE_LOG_LEVEL=INFO

# API
PODCAST_ENGINE_API_PORT=8000
PODCAST_ENGINE_API_WORKERS=2

# Kokoro TTS (Swarm internal network)
PODCAST_ENGINE_KOKORO_TTS_URL=http://serverlabapps-kokorotts-skwerq:8880/v1/audio/speech
PODCAST_ENGINE_KOKORO_TIMEOUT=120

# Storage (container paths)
PODCAST_ENGINE_STORAGE_BASE_PATH=/data/shared/podcasts
PODCAST_ENGINE_TEMP_DIR=/data/shared/podcasts/jobs
PODCAST_ENGINE_FINAL_DIR=/data/shared/podcasts/final

# Defaults
PODCAST_ENGINE_DEFAULT_VOICE=af_bella
PODCAST_ENGINE_DEFAULT_SPEED=1.0
PODCAST_ENGINE_DEFAULT_BITRATE=64k

# Processing
PODCAST_ENGINE_MAX_CONCURRENT_JOBS=3
PODCAST_ENGINE_MAX_PARALLEL_TTS_CALLS=5

# GUI
PODCAST_ENGINE_ENABLE_GUI=true
PODCAST_ENGINE_ENABLE_METRICS=true
```

---

## 📂 Étape 3 : Volumes (CRITIQUE)

Dans l'onglet **"Volumes"**, ajouter ce mount :

```
Host Path: /home/gilles/serverlab/data/shared
Container Path: /data/shared
Mode: Read/Write (rw)
```

**Important** : Ce volume est partagé avec n8n et autres services pour accès aux podcasts générés.

---

## 🌐 Étape 4 : Networking

### Port Mapping

```
Container Port: 8000
Host Port: 8000 (ou auto)
Protocol: TCP
```

### Domaine (optionnel mais recommandé)

```
Domain: podcast-engine.robotsinlove.be
SSL: Enabled (Let's Encrypt)
```

**Configuration Cloudflare** :
```
Type: A
Name: podcast-engine
Content: <IP_VPS>
Proxy: Orange Cloud (Proxied)
```

---

## 🔧 Étape 5 : Configuration avancée

### Health Check

```yaml
Test Command: curl -f http://localhost:8000/health || exit 1
Interval: 30s
Timeout: 10s
Retries: 3
Start Period: 15s
```

### Resources (optionnel)

```yaml
Memory Limit: 2GB
Memory Reservation: 512MB
CPU Limit: 2 cores
```

### Restart Policy

```yaml
Policy: unless-stopped
Max Attempts: 3
```

---

## 🚀 Étape 6 : Déploiement

1. **Vérifier configuration** :
   - ✅ Variables d'environnement toutes remplies
   - ✅ Volume `/data/shared` monté
   - ✅ Port 8000 exposé
   - ✅ Health check configuré

2. **Cliquer sur "Deploy"**

3. **Attendre le déploiement** (1-2 minutes)

4. **Vérifier les logs** :
   - Onglet "Logs" dans Dokploy
   - Chercher : `🚀 Starting Podcast Engine v1.0.0`
   - Chercher : `✓ Storage: /data/shared/podcasts`
   - Chercher : `✓ Kokoro TTS: http://serverlabapps-kokorotts-skwerq:8880`

---

## ✅ Étape 7 : Tests de validation

### Test 1 : Health Check

```bash
curl https://podcast-engine.robotsinlove.be/health
```

**Réponse attendue** :
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 123.45,
  "services": {
    "kokoro_tts": {
      "status": "healthy",
      "url": "http://serverlabapps-kokorotts-skwerq:8880"
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

### Test 2 : API Documentation

Ouvrir dans navigateur :
```
https://podcast-engine.robotsinlove.be/docs
```

Vérifier que Swagger UI s'affiche avec :
- ✅ `POST /api/v1/create-podcast`
- ✅ `GET /health`
- ✅ `GET /`

### Test 3 : GUI Web

Ouvrir dans navigateur :
```
https://podcast-engine.robotsinlove.be/gui
```

Vérifier que l'interface web s'affiche correctement.

### Test 4 : Créer un podcast test

```bash
curl -X POST https://podcast-engine.robotsinlove.be/api/v1/create-podcast \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ceci est un test du système de génération de podcasts. Il convertit automatiquement du texte en audio de haute qualité grâce à Kokoro TTS.",
    "metadata": {
      "title": "Test Podcast",
      "author": "Gilles",
      "language": "fr"
    },
    "tts_options": {
      "voice": "af_bella",
      "speed": 1.0
    },
    "processing_options": {
      "return_binary": false
    }
  }'
```

**Réponse attendue** (JSON avec podcast info).

---

## 🐛 Troubleshooting

### Problème : "Kokoro TTS unreachable"

**Cause** : Service Kokoro pas dans le même réseau Swarm

**Solution** :
```bash
# Vérifier que Kokoro tourne
docker service ls | grep kokoro

# Vérifier le réseau
docker service inspect serverlabapps-kokorotts-skwerq --format '{{json .Spec.TaskTemplate.Networks}}'

# S'assurer que podcast-engine est dans le même réseau
```

### Problème : "Storage not available"

**Cause** : Volume `/data/shared` mal monté

**Solution** :
```bash
# Vérifier sur host
ls -la /home/gilles/serverlab/data/shared/podcasts/

# Vérifier dans container
docker exec <container-id> ls -la /data/shared/podcasts/

# Fix permissions si nécessaire
sudo chmod -R 777 /home/gilles/serverlab/data/shared/podcasts/
```

### Problème : "ffmpeg not found"

**Cause** : Image Docker incomplète

**Solution** :
```bash
# Rebuild image
docker build -t podcast-engine:latest .

# Vérifier ffmpeg dans image
docker run --rm podcast-engine:latest ffmpeg -version
```

---

## 📊 Monitoring

### Logs en temps réel

```bash
# Via Docker
docker service logs serverlabapps-podcast-engine --follow

# Via Dokploy GUI
Onglet "Logs" → Enable "Auto-refresh"
```

### Métriques Prometheus (si activé)

```
http://podcast-engine.robotsinlove.be:9090/metrics
```

---

## 🔄 Mises à jour

Pour déployer une nouvelle version :

```bash
# 1. Rebuild image
docker build -t podcast-engine:latest .

# 2. Dans Dokploy GUI
#    - Onglet "General"
#    - Cliquer "Redeploy"
#    - Attendre nouveau déploiement

# 3. Vérifier logs
#    - Check version number dans logs startup
```

---

## 📝 Checklist post-déploiement

- [ ] Health check passe (status: healthy)
- [ ] Kokoro TTS accessible depuis container
- [ ] Volume `/data/shared` accessible
- [ ] GUI web affiche correctement
- [ ] API Docs (Swagger) accessible
- [ ] Test podcast créé avec succès
- [ ] Domaine DNS configuré (si applicable)
- [ ] SSL activé (Let's Encrypt)
- [ ] Logs pas d'erreurs critiques

---

**Status actuel** : Image buildée ✅ - Prêt pour déploiement Dokploy

**Next step** : Suivre ce guide dans Dokploy GUI
