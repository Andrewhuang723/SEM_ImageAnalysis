#!/bin/bash
# Deployment script for SEM Analysis Dashboard

echo "🚀 Deploying SEM Image Analysis Dashboard..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📝 Initializing git repository..."
    git init
fi

# Add all files
echo "📦 Adding files to git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Deploy SEM Analysis Dashboard - $(date)"

# Check if Heroku remote exists
if git remote | grep -q heroku; then
    echo "🔄 Deploying to existing Heroku app..."
    git push heroku main
else
    echo "🆕 Creating new Heroku app..."
    echo "Enter your desired app name (or press Enter for random name):"
    read app_name
    
    if [ -z "$app_name" ]; then
        heroku create
    else
        heroku create $app_name
    fi
    
    echo "🚀 Deploying to Heroku..."
    git push heroku main
fi

echo "✅ Deployment complete!"
echo "🌐 Your app should be available at: https://$(heroku apps:info --json | jq -r '.app.web_url')"
