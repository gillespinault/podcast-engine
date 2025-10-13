# GitHub Actions & GHCR Setup - Complete Guide

**Pour les futurs projets avec CI/CD automatique vers GHCR**

---

## 🎯 Problème Résolu

Le `GITHUB_TOKEN` automatique dans GitHub Actions a des **permissions insuffisantes** pour push vers GHCR (GitHub Container Registry) dans certaines configurations de compte.

**Erreur typique** :
```
ERROR: failed to push ghcr.io/user/image:tag
unexpected status from HEAD request: 403 Forbidden
```

---

## ✅ Solution Éprouvée : Personal Access Token

### Étape 1 : Créer un Personal Access Token

1. Va sur https://github.com/settings/tokens
2. Clique **"Generate new token (classic)"**
3. Nom : `GHCR_PUSH_TOKEN` (ou descriptif)
4. **Scopes REQUIS** (cocher obligatoirement) :
   - ☑️ `repo` (Full control of private repositories)
   - ☑️ `workflow` (Update GitHub Action workflows)
   - ☑️ `write:packages` (Upload packages to GHCR)
   - ☑️ `delete:packages` (Delete packages from GHCR)
   - ☑️ `read:packages` (Download packages from GHCR)
5. Génère et **copie le token immédiatement** (tu ne le reverras plus)

**Format token** : `ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`

### Étape 2 : Ajouter le Token dans les Secrets du Repository

1. Va sur `https://github.com/<user>/<repo>/settings/secrets/actions`
2. Clique **"New repository secret"**
3. Name : `GHCR_TOKEN`
4. Value : Colle le token généré
5. **"Add secret"**

### Étape 3 : Configurer le Workflow Docker

Fichier : `.github/workflows/docker-publish.yml`

```yaml
name: Docker Image CI/CD

on:
  push:
    branches: [ main ]
    paths:
      - 'app/**'
      - 'requirements.txt'
      - 'Dockerfile'
      - '.github/workflows/docker-publish.yml'
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: <username>/<project-name>  # ← MODIFIER ICI

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GHCR_TOKEN }}  # ← UTILISER LE SECRET

      - name: Extract metadata (tags, labels)
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### Étape 4 : Configurer les Permissions Repository

1. Va sur `https://github.com/<user>/<repo>/settings/actions`
2. Section **"Workflow permissions"**
3. Sélectionne : ☑️ **"Read and write permissions"**
4. Coche : ☑️ **"Allow GitHub Actions to create and approve pull requests"**
5. **"Save"**

---

## 🐳 Utilisation de l'Image GHCR dans Dokploy

### Ajouter les Credentials GHCR

**Dokploy → Settings → Registry** :

```
Type: GitHub Container Registry
Registry URL: ghcr.io
Username: <github-username>
Token/Password: <GHCR_TOKEN> (le même token)
```

### Déployer l'Application

**Dokploy → Create Application** :

```
Name: podcast-engine
Provider: Docker
Image: ghcr.io/<username>/<project-name>:latest
Pull Policy: Always (pour auto-update)
Registry: Sélectionner le registry GHCR configuré
```

**Environment Variables** (exemple pour Podcast Engine) :
```bash
PODCAST_ENGINE_API_KEY=<secret-key>
PODCAST_ENGINE_KOKORO_TTS_URL=http://kokorotts:8880/v1/audio/speech
```

**Volumes** :
```
/data/shared/podcasts → /data/shared/podcasts
```

---

## 🔄 Workflow de Développement

### Push Code → Auto Deploy

```bash
# 1. Développer en local
git add .
git commit -m "feat: nouvelle fonctionnalité"

# 2. Push vers GitHub
git push origin main

# 3. GitHub Actions se déclenche automatiquement
#    - Build Docker image (5-8 min)
#    - Push vers ghcr.io/<username>/<project>:latest
#    - Tags: latest, main, main-<commit-sha>

# 4. Dokploy pull automatiquement (si watchtower activé)
#    Ou redéployer manuellement dans Dokploy UI
```

---

## 🐛 Troubleshooting

### Erreur 403 Forbidden persistante

**Cause** : Package GHCR existant créé par un autre repository

**Solution** :
```bash
# Supprimer l'ancien package
TOKEN="<ton-token>"
curl -X DELETE \
  -H "Authorization: token $TOKEN" \
  "https://api.github.com/user/packages/container/<package-name>"

# Re-run le workflow
```

### Erreur "Bad credentials"

**Cause** : Token sans les bons scopes

**Solution** : Régénérer un token avec TOUS les scopes listés ci-dessus

### Build réussi mais image non disponible

**Vérifier** :
```bash
# Lister les packages
curl -H "Authorization: token $TOKEN" \
  "https://api.github.com/users/<username>/packages?package_type=container"

# Vérifier les versions
curl -H "Authorization: token $TOKEN" \
  "https://api.github.com/users/<username>/packages/container/<package>/versions"
```

---

## 📋 Checklist Nouveau Projet

- [ ] Créer repository GitHub
- [ ] Créer Personal Access Token avec scopes complets
- [ ] Ajouter secret `GHCR_TOKEN` dans repository
- [ ] Configurer workflow `.github/workflows/docker-publish.yml`
- [ ] Configurer permissions repository "Read and write"
- [ ] Push code → Vérifier build réussi
- [ ] Ajouter registry GHCR dans Dokploy
- [ ] Déployer application depuis GHCR

---

## 🎓 Leçons Apprises (Podcast Engine)

1. **GITHUB_TOKEN insuffisant** : Toujours utiliser PAT pour GHCR
2. **Permissions repository** : "Read and write" obligatoire
3. **Package ownership** : Un package GHCR = un repository source
4. **Token scopes** : `workflow` + `write:packages` critiques
5. **Workflow triggers** : Spécifier les paths pour éviter builds inutiles

---

**Dernière mise à jour** : 2025-10-13
**Testé sur** : podcast-engine v1.1.0
**Temps total setup** : ~20-30 min (si doc suivie)
