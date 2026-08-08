import zipfile
import xml.etree.ElementTree as ET

# PPTX is a ZIP file containing XML files
with zipfile.ZipFile('5深度学习.pptx', 'r') as z:
    # List all files in the PPTX
    print("=== PPTX 文件结构 ===")
    for name in z.namelist():
        print(name)
    
    print("\n\n=== 幻灯片内容 ===")
    # Find slide XML files
    slide_files = sorted([f for f in z.namelist() if f.startswith('ppt/slides/slide')])
    
    for slide_file in slide_files:
        print(f"\n--- {slide_file} ---")
        content = z.read(slide_file).decode('utf-8')
        # Extract text from XML
        root = ET.fromstring(content)
        # Namespace for PowerPoint
        ns = {
            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
            'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
        }
        # Find all text elements
        for t in root.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
            if t.text:
                print(t.text)
