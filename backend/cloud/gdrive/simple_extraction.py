#!/usr/bin/env python3
"""
Simple Google Drive File Extraction
Dead simple approach: if filename.pdf/doc/docx appears in the HTML, it's a file!
"""

import requests
import re
from typing import List, Dict, Set

class SimpleGoogleDriveExtractor:
    """Simple, reliable file extractor for Google Drive public folders."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def get_folder_page(self, folder_id: str) -> str:
        """Get the HTML content of a Google Drive folder page."""
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        try:
            print(f"🌐 Fetching folder: {folder_id}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"❌ Failed to fetch folder: {e}")
            return ""
    
    def extract_files_simple(self, html_content: str) -> List[Dict]:
        """
        SIMPLE EXTRACTION LOGIC:
        1. Find all strings that look like: filename.pdf, filename.doc, filename.docx
        2. Strict validation to only get real document files
        3. Extract clean file IDs for download
        """
        files = []
        found_filenames = set()
        
        print("🔍 SIMPLE FILE EXTRACTION:")
        print("   Looking for .pdf, .doc, .docx files in HTML...")
        
        # Simple regex: find any word.extension pattern
        file_patterns = [
            r'([a-zA-Z0-9_-]+\.pdf)',      # filename.pdf (NO dots in name)
            r'([a-zA-Z0-9_-]+\.docx?)',    # filename.doc or filename.docx (NO dots in name)
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            print(f"   📋 Pattern '{pattern}': {len(matches)} raw matches")
            
            for filename in matches:
                # Clean up the filename
                clean_filename = filename.strip()
                
                # Skip if already found (avoid duplicates)
                if clean_filename.lower() in found_filenames:
                    continue
                
                # STRICT validation - only real document files
                if self._is_valid_filename(clean_filename):
                    found_filenames.add(clean_filename.lower())
                    
                    # Extract clean file ID
                    file_id = self._extract_file_id_for_filename(html_content, clean_filename)
                    
                    # Skip if we couldn't find a valid file ID
                    if file_id.startswith('FALLBACK_'):
                        print(f"      ⚠️  Skipping {clean_filename} - no valid file ID found")
                        continue
                    
                    files.append({
                        'id': file_id,
                        'name': clean_filename,
                        'method': 'simple_extraction',
                        'download_url': f"https://drive.google.com/uc?export=download&id={file_id}"
                    })
                    print(f"      ✅ {clean_filename} (ID: {file_id[:15]}...)")
        
        return files
    
    def _is_valid_filename(self, filename: str) -> bool:
        """
        STRICT validation for document filenames.
        
        Rules:
        1. Must end with .pdf, .doc, or .docx
        2. No spaces in filename
        3. Only letters, numbers, underscore, dash allowed
        4. Only ONE dot (for the extension)
        5. No system/generic names
        6. No prefixes like 'x22' or weird characters
        """
        if not filename or len(filename) < 5:
            return False
        
        filename_lower = filename.lower()
        
        # Must end with our target extensions
        valid_extensions = ['.pdf', '.doc', '.docx']
        has_valid_extension = any(filename_lower.endswith(ext) for ext in valid_extensions)
        
        if not has_valid_extension:
            return False
        
        # No spaces (as per user specification)
        if ' ' in filename:
            return False
        
        # STRICT: Only ONE dot (for extension), no additional dots
        dot_count = filename.count('.')
        if dot_count != 1:
            return False
        
        # STRICT: Only letters, numbers, underscore, dash (NO additional dots)
        if not re.match(r'^[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+$', filename):
            return False
        
        # No weird prefixes or encoded characters
        if filename.startswith(('x22', 'x2', '%', 'vnd.', 'window.', 'document.', '_.', 'g.')):
            return False
        
        # No system file types or MIME type names
        system_names = [
            'google-apps', 'wordprocessingml', 'ms-word', 'openxml', 
            'officedocument', 'google-gsuite', 'application', 'mime'
        ]
        name_part = filename.rsplit('.', 1)[0].lower()  # Get name without extension
        if any(sys_name in name_part for sys_name in system_names):
            return False
        
        # Skip very short names or single character names
        name_part = filename.rsplit('.', 1)[0]  # Get name without extension
        if len(name_part) < 3:
            return False
        
        # Skip system/generic names
        skip_patterns = [
            'document.pdf', 'file.pdf', 'untitled.pdf', 'temp.pdf', 
            'test.pdf', 'sample.pdf', 'a.doc', 'c.doc', 'b.doc'
        ]
        
        if filename_lower in skip_patterns:
            return False
        
        return True
    
    def _clean_file_id(self, file_id: str) -> str:
        """
        Clean up Google Drive file ID by removing prefixes and invalid characters.
        
        Args:
            file_id: Raw file ID extracted from HTML
            
        Returns:
            Clean file ID suitable for Google Drive downloads
        """
        # Remove common prefixes
        if file_id.startswith('x22'):
            file_id = file_id[3:]  # Remove 'x22' prefix
        elif file_id.startswith('x2'):
            file_id = file_id[2:]  # Remove 'x2' prefix
        
        # Google Drive file IDs should be alphanumeric + dash/underscore
        # Keep only valid characters
        clean_id = re.sub(r'[^a-zA-Z0-9_-]', '', file_id)
        
        # Must be at least 25 characters for a valid Google Drive ID
        if len(clean_id) >= 25:
            return clean_id
        
        # If cleaning made it too short, return original
        return file_id
    
    def _extract_file_id_for_filename(self, html_content: str, filename: str) -> str:
        """
        Extract file ID for a specific filename.
        
        SIMPLE APPROACH:
        1. Find where the filename appears in the HTML
        2. Look for Google Drive file IDs nearby (28-44 character alphanumeric strings)
        3. Return the most likely one
        """
        # Find all positions where this filename appears
        positions = []
        start = 0
        while True:
            pos = html_content.find(filename, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        print(f"      📍 {filename} found at {len(positions)} positions")
        
        # For each position, look for file IDs nearby
        for pos in positions:
            # Get context around the filename (±500 characters)
            context_start = max(0, pos - 500)
            context_end = min(len(html_content), pos + 500)
            context = html_content[context_start:context_end]
            
            # Look for Google Drive file IDs (typically 28-44 characters)
            id_pattern = r'([a-zA-Z0-9_-]{28,44})'
            ids = re.findall(id_pattern, context)
            
            # Return the first reasonable ID (prefer longer ones)
            for file_id in sorted(ids, key=len, reverse=True):
                if len(file_id) >= 28:
                    # Clean up file ID - remove prefixes like 'x22'
                    clean_id = self._clean_file_id(file_id)
                    print(f"         📄 Found ID: {clean_id[:20]}...")
                    return clean_id
        
        # Fallback: generate a deterministic ID based on filename
        # This won't work for downloads but helps with testing
        import hashlib
        fallback_id = hashlib.md5(filename.encode()).hexdigest()
        print(f"         🔧 Generated fallback ID: {fallback_id[:20]}...")
        return f"FALLBACK_{fallback_id}"
    
    def test_simple_extraction(self, folder_id: str = "YOUR-FOLDER-ID-HERE") -> Dict:
        """Test the simple extraction logic."""
        print(f"🎯 Testing SIMPLE extraction on folder: {folder_id}")
        
        html_content = self.get_folder_page(folder_id)
        if not html_content:
            return {'success': False, 'error': 'Failed to fetch page'}
        
        print(f"📄 Page size: {len(html_content)} characters")
        
        # Extract files using simple method
        files = self.extract_files_simple(html_content)
        
        print(f"\n✅ EXTRACTION RESULTS:")
        print(f"   📊 Total files detected: {len(files)}")
        
        for i, file_info in enumerate(files, 1):
            print(f"   {i}. {file_info['name']}")
            print(f"      ID: {file_info['id'][:30]}...")
            print(f"      Download: {file_info['download_url'][:50]}...")
        
        # Check against expected files
        expected_files = [
            'credit_rating_report_page_1.pdf',
            'credit_rating_report_sector.pdf'
        ]
        
        print(f"\n🎯 VERIFICATION:")
        all_found = True
        for expected in expected_files:
            found = any(expected.lower() == f['name'].lower() for f in files)
            status = "✅ FOUND" if found else "❌ MISSING"
            print(f"   {expected}: {status}")
            if not found:
                all_found = False
        
        return {
            'success': True,
            'files_found': len(files),
            'files': files,
            'expected_count': len(expected_files),
            'all_expected_found': all_found
        }

if __name__ == "__main__":
    print("🚀 Simple Google Drive File Extraction Test")
    print("🎯 Philosophy: If filename.pdf is in the HTML, it's a file!")
    
    extractor = SimpleGoogleDriveExtractor()
    result = extractor.test_simple_extraction()
    
    if result['success']:
        print(f"\n" + "="*60)
        print(f"📊 FINAL SUMMARY")
        print(f"="*60)
        print(f"✅ Files detected: {result['files_found']}")
        print(f"🎯 Expected files: {result['expected_count']}")
        
        if result['all_expected_found']:
            print(f"🎉 SUCCESS: All expected files detected!")
            print(f"🚀 This approach works - ready for integration!")
        else:
            print(f"📝 Files found: {[f['name'] for f in result['files']]}")
            print(f"🔍 May need minor adjustments to catch all files")
    else:
        print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
