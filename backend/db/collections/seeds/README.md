# Database Seed Files

This directory contains MongoDB seed data for the Document Intelligence demo.

## Security Notice

⚠️ **IMPORTANT**: Do not commit files containing real S3 bucket names or Google Drive folder IDs to public repositories.

## Files

### Safe to Commit
- `document_intelligence.industry_mappings.json` - Industry and topic classifications (no sensitive data)
- `*.example.json` - Template files with placeholder values

### Not Included in Repository
- `document_intelligence.buckets.json` - Would contain real S3 bucket configurations
- `document_intelligence.gdrive.json` - Would contain real Google Drive folder IDs

## Setting Up Your Environment

1. Copy the example files:
   ```bash
   cp document_intelligence.buckets.example.json document_intelligence.buckets.json
   cp document_intelligence.gdrive.example.json document_intelligence.gdrive.json
   ```

2. Replace placeholder values with your actual configurations:
   - S3 bucket names and paths
   - Google Drive folder IDs
   - MongoDB Object IDs (or generate new ones)

3. Import seed data:
   ```bash
   # Import all seed data
   mongoimport --uri "$MONGODB_URI" --collection buckets --file document_intelligence.buckets.json --jsonArray
   mongoimport --uri "$MONGODB_URI" --collection gdrive --file document_intelligence.gdrive.json --jsonArray
   mongoimport --uri "$MONGODB_URI" --collection industry_mappings --file document_intelligence.industry_mappings.json --jsonArray
   ```

## Why This Matters

Even though S3 buckets and Google Drive folders might be publicly accessible:
- Exposing bucket names makes them targets for abuse (cost attacks, enumeration)
- Public repositories are scraped by bots looking for AWS resources
- It's a security best practice to never commit infrastructure details

## For Demo Purposes

If you need to share this demo:
1. Use the example files as templates
2. Document that users need to set up their own S3 buckets and Google Drive folders
3. Provide instructions on how to configure the system with their own resources
