# DocumentViz - Intelligent Document Visualization Platform

Transform your business documents into professional visualizations using AI-powered analysis and customizable templates.

## 🚀 Features

- **AI-Powered Analysis**: Uses Google Gemini 2.5-Flash for intelligent content extraction
- **Multiple Document Types**: Supports PDF, DOCX, TXT, and JSON formats
- **Professional Templates**: Pre-built templates for value propositions, competitive analysis, product roadmaps, and risk analysis
- **PNG/JPEG Export**: High-quality image downloads for presentations and reports
- **Real-time Processing**: Fast document analysis with caching for improved performance
- **Interactive Web Interface**: Clean, responsive UI with drag-and-drop file uploads

## 📋 Supported Visualization Types

### Value Proposition Analysis
- Customer segments and pain points
- Solution benefits and differentiation
- Market positioning and competitive advantages

### Competitive Analysis
- Feature comparison matrix
- Market positioning analysis
- Pricing and support comparison
- Limitations assessment

### Product Roadmap
- Timeline visualization
- Phase-based planning
- Deliverable tracking
- Milestone management

### Risk Analysis
- Risk identification and assessment
- Impact analysis
- Mitigation strategies
- Risk matrix visualization

## 🛠️ Technology Stack

- **Backend**: Python 3.11 with FastAPI
- **AI/ML**: Google Gemini API, spaCy, scikit-learn
- **Document Processing**: python-docx, PyPDF2
- **Visualization**: Dynamic SVG generation with PNG/JPEG export
- **Frontend**: HTML/CSS/JavaScript with Tailwind CSS
- **Database**: SQLAlchemy with file-based persistence

## 🏃‍♂️ Quick Start

### Prerequisites
- Python 3.11 or higher
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Codachriss/CopyWorkingvisual-1.git
cd CopyWorkingvisual-1

Install dependencies:
pip install -r requirements.txt
Set up your API key:
export GEMINI_API_KEY="your-api-key-here"
Run the application:
python documentviz.py --port 5000 --host 0.0.0.0

Open your browser and navigate to http://localhost:5000
📖 Usage
Upload Document: Drag and drop or select a business document (PDF, DOCX, TXT, JSON)
AI Analysis: The system automatically analyzes content and determines document type
Generate Visualization: Creates a professional visualization based on extracted content
Download: Export as PNG/JPEG for presentations or reports
🎯 Use Cases
Product Managers: Create professional roadmaps and competitive analysis
Business Analysts: Generate value proposition and risk assessment visuals
Consultants: Transform client documents into presentation-ready graphics
Startups: Quickly create investor-ready business visualizations
🔧 Configuration
Environment Variables
GEMINI_API_KEY: Your Google Gemini API key
PORT: Application port (default: 5000)
HOST: Host address (default: 0.0.0.0)
Template Customization
Templates are located in the templates/ directory and can be customized:

templates/value_proposition/
templates/competitive_analysis/
templates/product_roadmap/
templates/risk_analysis/
📊 Output Formats
PNG: High-resolution raster images (recommended for presentations)
JPEG: Compressed images for web use
SVG: Vector graphics for scalability (internal use)
🔒 Privacy & Security
Documents are processed locally
No data is stored permanently after processing
API calls to Gemini are made securely
File uploads are validated and sanitized
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

📞 Support
For support, please open an issue on GitHub or contact the maintainers.

Built with ❤️ for better business document visualization
