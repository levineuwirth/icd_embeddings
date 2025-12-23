#!/bin/bash
# Deployment script for Mac Studio with Docker + Cloudflare Tunnel

set -e

echo "🚀 ICD Prediction Website - Docker Deployment"
echo "=============================================="

# Step 1: Build frontend with Docker environment
echo ""
echo "📦 Step 1: Building frontend..."
cp .env.docker .env
npm install
npm run build
echo "✅ Frontend built to ./dist/"

# Step 2: Build and start Docker containers
echo ""
echo "🐳 Step 2: Building Docker images..."
docker-compose build --no-cache

echo ""
echo "🚀 Step 3: Starting containers..."
docker-compose up -d

# Step 4: Wait for services to be healthy
echo ""
echo "⏳ Step 4: Waiting for services to be healthy..."
sleep 5

# Check if containers are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Containers are running"
    docker-compose ps
else
    echo "❌ Error: Containers failed to start"
    docker-compose logs
    exit 1
fi

# Step 5: Test the deployment
echo ""
echo "🧪 Step 5: Testing deployment..."
if curl -f http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "⚠️  Warning: Health check failed (services may still be starting up)"
fi

echo ""
echo "=============================================="
echo "✨ Deployment complete!"
echo ""
echo "Services:"
echo "  • Backend: http://127.0.0.1:8080/api/"
echo "  • Frontend: http://127.0.0.1:8080/"
echo "  • Health: http://127.0.0.1:8080/health"
echo ""
echo "Next steps:"
echo "  1. Test locally: open http://127.0.0.1:8080"
echo "  2. Configure cloudflared tunnel to point to 127.0.0.1:8080"
echo "  3. Monitor logs: docker-compose logs -f"
echo ""
echo "Commands:"
echo "  • Stop:    docker-compose down"
echo "  • Restart: docker-compose restart"
echo "  • Logs:    docker-compose logs -f [service]"
echo "  • Update:  ./deploy.sh"
echo "=============================================="
