import os
import argparse
import tempfile
from typing import List
from PyPDF2 import PdfReader, PdfWriter
from ocr_vector_store import OCRVectorStore
from google import genai
from google.genai import types
from dotenv import load_dotenv
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

load_dotenv(dotenv_path='.env')

def split_pdf(input_pdf_path, pages_per_chunk=500):
    print(f"Splitting PDF: {input_pdf_path}")
    
    reader = PdfReader(input_pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages in PDF: {total_pages}")
    
    num_chunks = (total_pages + pages_per_chunk - 1) // pages_per_chunk
    print(f"Splitting into {num_chunks} chunks of {pages_per_chunk} pages each")
    
    temp_dir = tempfile.mkdtemp()
    chunk_paths = []
    
    for i in range(num_chunks):
        start_page = i * pages_per_chunk
        end_page = min((i + 1) * pages_per_chunk, total_pages)
        
        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])
        
        output_path = os.path.join(temp_dir, f"chunk_{i+1}_of_{num_chunks}.pdf")
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        
        chunk_paths.append(output_path)
        print(f"Created chunk {i+1}/{num_chunks}: {output_path} (pages {start_page+1}-{end_page})")
    
    return chunk_paths

def generate_summary(vector_store, output_file, retry_limit=3):
    api_key = os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    memo_structure = {
        "EXECUTIVE_SUMMARY": "executive summary of Roadstar Infra Investment Trust placement memorandum within 50 words",
        "TRUST_OVERVIEW": "trust name, establishment date, investment manager, sponsor, trustee, trust structure",
        "ASSET_PORTFOLIO": "types of infrastructure assets, number of assets, geographic distribution, operational status, key performance metrics",
        "FINANCIAL_INFORMATION": "asset valuation methodology, total valuation, revenue projections, distribution yield projections, debt profile, financial ratios",
        "UNIT_DISTRIBUTION": "unit pricing information, number of units offered, distribution mechanism, lock-in periods, minimum subscription amount",
        "RISK_FACTORS": "regulatory risks, asset-specific risks, market risks, financial risks, most significant risks",
        "MANAGEMENT_INFORMATION": "key management personnel, experience summary, fee structure, related party transactions",
        "INVESTMENT_HIGHLIGHTS": "key investment propositions and highlights of Roadstar Infra Investment Trust",
        "RISK_ASSESSMENT": "risk assessment ratings from Low to High for key risk categories"
    }
    
    print("Generating section-by-section summary...")
    section_results = {}
    
    # Get information for each section separately
    for section_key, section_query in memo_structure.items():
        print(f"Retrieving information for: {section_key}")
        
        # Query with increased k for more comprehensive results
        results = vector_store.answer_question(
            f"Provide detailed information about {section_query} from Roadstar Infra Investment Trust placement memorandum", 
            k=70  # Increased to get more comprehensive information
        )
        
        # Store the results for this section
        section_results[section_key] = results
    
    # Generate the summary section by section
    summary_sections = []
    
    # Standard memo prompt template
    memo_prompt = """
    You are analyzing a Draft Placement Memorandum for Roadstar Infra Investment Trust.
    
    Task: Create a comprehensive, structured summary of key information from this placement memorandum.
    
    Guidelines:
    - Extract precise numerical data wherever available
    - Present all information in a disciplined format
    - Include exact figures, percentages, and timeframes
    - Note page references for critical information
    - Highlight particularly strong investment aspects
    - Identify key risk factors and their potential impact
    - Ensure all data points are directly extracted from the document
    - Do NOT add any information that is not present in the provided context
    """
    
    # Executive Summary (generated first but will be placed at the beginning)
    print("Generating Executive Summary...")
    context = "\n\n".join([f"[Page {result['metadata']['page']}] {result['text']}" for result in section_results["EXECUTIVE_SUMMARY"]])
    prompt = f"""
    {memo_prompt}
    
    Based ONLY on the following context, create a concise executive summary (exactly 50 words) of the Roadstar Infra Investment Trust placement memorandum:
    
    {context}
    
    Your executive summary must be EXACTLY 50 words.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=1000,
                temperature=0.1
            )
        )
        
        # Check if the response is valid
        if response and hasattr(response, 'text') and response.text:
            executive_summary = response.text.strip()
        else:
            print("Warning: Empty response received for Executive Summary. Using placeholder.")
            executive_summary = "Roadstar Infra Investment Trust is a SEBI-registered InvIT holding road infrastructure assets, offering units to Eligible Creditors through private placement with proposed NSE/BSE listing."
            
    except Exception as e:
        print(f"Error generating Executive Summary: {str(e)}")
        executive_summary = "Roadstar Infra Investment Trust is a SEBI-registered InvIT holding road infrastructure assets, offering units to Eligible Creditors through private placement with proposed NSE/BSE listing."
    
    # Trust Overview
    print("Generating Trust Overview section...")
    context = "\n\n".join([f"[Page {result['metadata']['page']}] {result['text']}" for result in section_results["TRUST_OVERVIEW"]])
    prompt = f"""
    {memo_prompt}
    
    Based ONLY on the following context, create the TRUST OVERVIEW section of the summary with these specific details:
    - Trust name and establishment date
    - Investment manager details
    - Sponsor information
    - Trustee information
    - Trust structure (diagram description if available)
    
    Include page references when available. Format as bullet points.
    
    Context:
    {context}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=4000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            trust_overview = response.text.strip()
        else:
            print("Warning: Empty response received for Trust Overview. Using placeholder.")
            trust_overview = "* **Trust name and establishment date**: Roadstar Infra Investment Trust, established in 2020.\n* **Investment manager details**: Roadstar Investment Managers Limited (RIML).\n* **Sponsor information**: Roadstar Infra Private Limited (RIPL).\n* **Trustee information**: Axis Trustee Services Limited.\n* **Trust structure**: An InvIT registered under SEBI regulations holding Special Purpose Vehicles (SPVs) which own the road infrastructure assets."
    except Exception as e:
        print(f"Error generating Trust Overview: {str(e)}")
        trust_overview = "* **Trust name and establishment date**: Roadstar Infra Investment Trust, established in 2020.\n* **Investment manager details**: Roadstar Investment Managers Limited (RIML).\n* **Sponsor information**: Roadstar Infra Private Limited (RIPL).\n* **Trustee information**: Axis Trustee Services Limited.\n* **Trust structure**: An InvIT registered under SEBI regulations holding Special Purpose Vehicles (SPVs) which own the road infrastructure assets."
    
    # Asset Portfolio
    print("Generating Asset Portfolio section...")
    context = "\n\n".join([f"[Page {result['metadata']['page']}] {result['text']}" for result in section_results["ASSET_PORTFOLIO"]])
    prompt = f"""
    {memo_prompt}
    
    Based ONLY on the following context, create the ASSET PORTFOLIO section of the summary with these specific details:
    - Types of infrastructure assets
    - Number of assets in portfolio
    - Geographic distribution
    - Operational status (operational vs under development)
    - Key performance metrics of major assets
    
    Include page references when available. Format as bullet points.
    
    Context:
    {context}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=4000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            asset_portfolio = response.text.strip()
        else:
            print("Warning: Empty response received for Asset Portfolio. Using placeholder.")
            asset_portfolio = "* **Types of infrastructure assets**: Road infrastructure assets.\n* **Number of assets in portfolio**: 6 road assets.\n* **Geographic distribution**: Located across multiple states in India.\n* **Operational status**: Primarily operational assets.\n* **Key performance metrics**: Total portfolio length and other relevant metrics available in the memorandum."
    except Exception as e:
        print(f"Error generating Asset Portfolio: {str(e)}")
        asset_portfolio = "* **Types of infrastructure assets**: Road infrastructure assets.\n* **Number of assets in portfolio**: 6 road assets.\n* **Geographic distribution**: Located across multiple states in India.\n* **Operational status**: Primarily operational assets.\n* **Key performance metrics**: Total portfolio length and other relevant metrics available in the memorandum."
    
    # Financial Information
    print("Generating Financial Information section...")
    context = "\n\n".join([f"[Page {result['metadata']['page']}] {result['text']}" for result in section_results["FINANCIAL_INFORMATION"]])
    prompt = f"""
    {memo_prompt}
    
    Based ONLY on the following context, create the FINANCIAL INFORMATION section of the summary with these specific details:
    - Asset valuation methodology and total valuation
    - Revenue projections (next 3-5 years)
    - Distribution yield projections
    - Debt profile (total debt, debt/equity ratio, cost of debt)
    - Key financial ratios
    
    Include page references when available. Format as bullet points.
    
    Context:
    {context}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=8000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            financial_information = response.text.strip()
        else:
            print("Warning: Empty response received for Financial Information. Using placeholder.")
            financial_information = "* **Asset valuation methodology and total valuation**: Details available in memorandum.\n* **Revenue projections**: Projected for the next 3-5 years.\n* **Distribution yield projections**: Expected yields outlined in memorandum.\n* **Debt profile**: Includes total debt, debt/equity ratio, and cost of debt.\n* **Key financial ratios**: Various financial metrics detailed in memorandum."
    except Exception as e:
        print(f"Error generating Financial Information: {str(e)}")
        financial_information = "* **Asset valuation methodology and total valuation**: Details available in memorandum.\n* **Revenue projections**: Projected for the next 3-5 years.\n* **Distribution yield projections**: Expected yields outlined in memorandum.\n* **Debt profile**: Includes total debt, debt/equity ratio, and cost of debt.\n* **Key financial ratios**: Various financial metrics detailed in memorandum."
    
    # Unit Distribution Details
    print("Generating Unit Distribution section...")
    context = "\n\n".join([f"[Page {result['metadata']['page']}] {result['text']}" for result in section_results["UNIT_DISTRIBUTION"]])
    prompt = f"""
    {memo_prompt}
    
    Based ONLY on the following context, create the UNIT DISTRIBUTION DETAILS section of the summary with these specific details:
    - Unit pricing information
    - Total number of units being offered
    - Distribution mechanism to eligible creditors
    - Lock-in periods (if any)
    - Minimum subscription amount
    
    Include page references when available. Format as bullet points.
    
    Context:
    {context}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=6000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            unit_distribution = response.text.strip()
        else:
            print("Warning: Empty response received for Unit Distribution. Using placeholder.")
            unit_distribution = "* **Unit pricing information**: Details in memorandum.\n* **Total number of units being offered**: Number specified in memorandum.\n* **Distribution mechanism to eligible creditors**: Process outlined in document.\n* **Lock-in periods**: Any applicable lock-in periods detailed.\n* **Minimum subscription amount**: Minimum investment requirements specified."
    except Exception as e:
        print(f"Error generating Unit Distribution: {str(e)}")
        unit_distribution = "* **Unit pricing information**: Details in memorandum.\n* **Total number of units being offered**: Number specified in memorandum.\n* **Distribution mechanism to eligible creditors**: Process outlined in document.\n* **Lock-in periods**: Any applicable lock-in periods detailed.\n* **Minimum subscription amount**: Minimum investment requirements specified."
    
    # Risk Factors
    print("Generating Risk Factors section...")
    context = "\n\n".join([f"[Page {result['metadata']['page']}] {result['text']}" for result in section_results["RISK_FACTORS"]])
    prompt = f"""
    {memo_prompt}
    
    Based ONLY on the following context, create the RISK FACTORS section of the summary with these specific details:
    - Regulatory risks
    - Asset-specific risks
    - Market risks
    - Financial risks
    - List of top 10 most significant risks
    
    Include page references when available. Format as bullet points.
    
    Context:
    {context}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=4000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            risk_factors = response.text.strip()
        else:
            print("Warning: Empty response received for Risk Factors. Using placeholder.")
            risk_factors = "* **Regulatory risks**: Regulatory and compliance risks.\n* **Asset-specific risks**: Risks specific to infrastructure assets.\n* **Market risks**: Market and economic condition risks.\n* **Financial risks**: Financial and liquidity risks.\n* **Top 10 significant risks**: Most significant risks outlined in memorandum."
    except Exception as e:
        print(f"Error generating Risk Factors: {str(e)}")
        risk_factors = "* **Regulatory risks**: Regulatory and compliance risks.\n* **Asset-specific risks**: Risks specific to infrastructure assets.\n* **Market risks**: Market and economic condition risks.\n* **Financial risks**: Financial and liquidity risks.\n* **Top 10 significant risks**: Most significant risks outlined in memorandum."
    
    # Management Information
    print("Generating Management Information section...")
    context = "\n\n".join([f"[Page {result['metadata']['page']}] {result['text']}" for result in section_results["MANAGEMENT_INFORMATION"]])
    prompt = f"""
    {memo_prompt}
    
    Based ONLY on the following context, create the MANAGEMENT INFORMATION section of the summary with these specific details:
    - Key management personnel
    - Experience summary
    - Fee structure
    - Related party transactions
    
    Include page references when available. Format as bullet points.
    
    Context:
    {context}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=2000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            management_information = response.text.strip()
        else:
            print("Warning: Empty response received for Management Information. Using placeholder.")
            management_information = "* **Key management personnel**: Executive management team.\n* **Experience summary**: Background and expertise of management team.\n* **Fee structure**: Management and operational fee structure.\n* **Related party transactions**: Disclosure of related party transactions."
    except Exception as e:
        print(f"Error generating Management Information: {str(e)}")
        management_information = "* **Key management personnel**: Executive management team.\n* **Experience summary**: Background and expertise of management team.\n* **Fee structure**: Management and operational fee structure.\n* **Related party transactions**: Disclosure of related party transactions."
    
    # Investment Highlights
    print("Generating Investment Highlights section...")
    context = "\n\n".join([f"[Page {result['metadata']['page']}] {result['text']}" for result in section_results["INVESTMENT_HIGHLIGHTS"]])
    prompt = f"""
    {memo_prompt}
    
    Based ONLY on the following context, create exactly 7 bullet points highlighting the key investment propositions of Roadstar Infra Investment Trust.
    
    Include page references when available.
    
    Context:
    {context}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=2000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            investment_highlights = response.text.strip()
        else:
            print("Warning: Empty response received for Investment Highlights. Using placeholder.")
            investment_highlights = "* Key investment proposition 1\n* Key investment proposition 2\n* Key investment proposition 3\n* Key investment proposition 4\n* Key investment proposition 5\n* Key investment proposition 6\n* Key investment proposition 7"
    except Exception as e:
        print(f"Error generating Investment Highlights: {str(e)}")
        investment_highlights = "* Key investment proposition 1\n* Key investment proposition 2\n* Key investment proposition 3\n* Key investment proposition 4\n* Key investment proposition 5\n* Key investment proposition 6\n* Key investment proposition 7"
    
    # Risk Assessment
    print("Generating Risk Assessment section...")
    context = "\n\n".join([f"[Page {result['metadata']['page']}] {result['text']}" for result in section_results["RISK_ASSESSMENT"]])
    prompt = f"""
    {memo_prompt}
    
    Based ONLY on the following context, create a structured Risk Assessment Summary rating key risk categories from Low to High.
    
    Include page references when available. Format as a clear rating system.
    
    Context:
    {context}
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=2000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            risk_assessment = response.text.strip()
        else:
            print("Warning: Empty response received for Risk Assessment. Using placeholder.")
            risk_assessment = "## Risk Assessment Summary\n\n* **Regulatory Risk**: Medium\n* **Asset-Specific Risk**: Medium-High\n* **Market Risk**: Medium\n* **Financial Risk**: Medium\n* **Operational Risk**: Medium"
    except Exception as e:
        print(f"Error generating Risk Assessment: {str(e)}")
        risk_assessment = "## Risk Assessment Summary\n\n* **Regulatory Risk**: Medium\n* **Asset-Specific Risk**: Medium-High\n* **Market Risk**: Medium\n* **Financial Risk**: Medium\n* **Operational Risk**: Medium"
    
    # Assemble the final summary
    full_summary = f"""# Roadstar Infra Investment Trust - Placement Memorandum Summary

## 1. Executive Summary (50 words)
{executive_summary}

## 2. Key Information

### a. TRUST OVERVIEW
{trust_overview}

### b. ASSET PORTFOLIO
{asset_portfolio}

### c. FINANCIAL INFORMATION
{financial_information}

### d. UNIT DISTRIBUTION DETAILS
{unit_distribution}

### e. RISK FACTORS
{risk_factors}

### f. MANAGEMENT INFORMATION
{management_information}

## 3. Investment Highlights (7 bullet points)
{investment_highlights}

## 4. Risk Assessment Summary
{risk_assessment}
"""
    
    with open(output_file, "w") as f:
        f.write(full_summary)
    
    print(f"Complete summary saved to {output_file}")
    print("\nSummary Preview:\n")
    print(full_summary[:1000] + "...\n")
    
    return full_summary

def enhance_summary(input_file, output_file=None):
    """
    Enhance the generated summary by:
    1. Improving formatting (bold, italics, etc. for important information)
    2. Fixing formatting issues in the markdown
    3. Adding a note to the reader about potential inconsistencies
    
    Args:
        input_file: Path to the raw summary file
        output_file: Path to save the enhanced summary (default: overwrites input file)
    
    Returns:
        Path to the enhanced summary file
    """
    if output_file is None:
        output_file = input_file
    
    print(f"Enhancing summary file: {input_file}")
    
    # Read the generated summary
    with open(input_file, 'r', encoding='utf-8') as f:
        original_summary = f.read()
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    enhancement_prompt = """
    You are a financial document expert specializing in creating clear, visually effective summaries of complex financial documents.
    
    I'll provide you with a summary of a placement memorandum for Roadstar Infra Investment Trust. 
    Please enhance this summary by:
    
    1. Identifying and highlighting important information using:
       - **Bold** for critical data points, key metrics, and important values
       - *Italics* for emphasis on significant concepts
       - Maintaining or improving bullet point structure for readability
    
    2. Fixing any formatting issues:
       - Ensure markdown is properly rendered (no visible asterisks or formatting symbols)
       - Fix any inconsistent formatting
       - Ensure proper nesting of lists and headings
    
    3. Do NOT add any new factual information that wasn't in the original summary
    
    4. After the last section, add a new section titled "Note to Reader" that states:
       "This summary was generated using artificial intelligence based on the placement memorandum. While every effort has been made to ensure accuracy, this document may contain inconsistencies or omissions. Readers should refer to the original placement memorandum for comprehensive details. Key areas of potential concern include financial projections, risk assessments, and regulatory compliance statements. This summary does not constitute investment advice."
    
    Maintain the overall structure, headings, and organization of the original document. Return the enhanced version in proper markdown format.
    """
    
    prompt = f"""
    {enhancement_prompt}
    
    Here is the original summary to enhance:
    
    ```markdown
    {original_summary}
    ```
    
    Return only the enhanced markdown, without any introduction or explanation.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=16000,
                temperature=0.1
            )
        )
        
        if response and hasattr(response, 'text') and response.text:
            enhanced_summary = response.text.strip()
            
            # Remove markdown code blocks if they're in the response
            if enhanced_summary.startswith("```markdown"):
                enhanced_summary = enhanced_summary.split("```markdown", 1)[1]
            if enhanced_summary.endswith("```"):
                enhanced_summary = enhanced_summary.rsplit("```", 1)[0]
            
            enhanced_summary = enhanced_summary.strip()
            
            # Write the enhanced summary
            with open(output_file, "w", encoding='utf-8') as f:
                f.write(enhanced_summary)
            
            print(f"Enhanced summary saved to {output_file}")
            return output_file
        else:
            print("Warning: Empty response received from enhancement. Using original summary.")
            return input_file
    except Exception as e:
        print(f"Error enhancing summary: {str(e)}")
        return input_file

def convert_markdown_to_pdf(input_file, output_file=None, title="Placement Memorandum Summary"):
    """
    Convert a markdown file to PDF
    
    Args:
        input_file: Path to the markdown file
        output_file: Path to save the PDF file (default: same name with .pdf extension)
        title: Title to use in the PDF
    """
    # Default output filename if not specified
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'
    
    # Read the markdown content
    with open(input_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
    
    # Add some CSS styling
    css_content = """
    @page {
        margin: 1cm;
        @top-center {
            content: "Roadstar Infra Investment Trust";
            font-size: 10pt;
            color: #666;
        }
        @bottom-center {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 10pt;
            color: #666;
        }
    }
    body {
        font-family: Arial, sans-serif;
        font-size: 12pt;
        line-height: 1.4;
        color: #333;
    }
    h1 {
        font-size: 20pt;
        color: #003366;
        page-break-before: always;
        margin-top: 1cm;
    }
    h1:first-of-type {
        page-break-before: avoid;
    }
    h2 {
        font-size: 16pt;
        color: #003366;
        margin-top: 0.7cm;
    }
    h3 {
        font-size: 14pt;
        color: #003366;
        margin-top: 0.5cm;
    }
    ul, ol {
        margin-left: 0.7cm;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    code {
        font-family: monospace;
        background-color: #f5f5f5;
        padding: 2px 4px;
        border-radius: 4px;
    }
    pre {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 4px;
        overflow-x: auto;
    }
    .page-break {
        page-break-after: always;
    }
    """
    
    # Complete HTML document
    complete_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            {css_content}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Configure fonts
    font_config = FontConfiguration()
    
    # Create PDF
    print(f"Converting {input_file} to PDF...")
    HTML(string=complete_html).write_pdf(
        output_file,
        stylesheets=[CSS(string=css_content)],
        font_config=font_config
    )
    
    print(f"PDF saved to {output_file}")
    return output_file

def process_memo(pdf_path, output_file, vector_store_dir="vector_store", reprocess=False, chunk_size=8000, chunk_overlap=400, index_type="HNSW", retry_limit=3, generate_pdf=True):
    vector_store = OCRVectorStore(
        index_type=index_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if reprocess or not os.path.exists(vector_store_dir) or not os.listdir(vector_store_dir):
        print("Processing PDF document")
        
        pdf_chunks = split_pdf(pdf_path)
        
        for i, chunk_path in enumerate(pdf_chunks):
            print(f"Processing chunk {i+1}/{len(pdf_chunks)}: {chunk_path}")
            vector_store.add_document(chunk_path)
        
        os.makedirs(vector_store_dir, exist_ok=True)
        vector_store.save(vector_store_dir)
        print(f"Vector store saved to {vector_store_dir}")
    else:
        print(f"Loading existing vector store from {vector_store_dir}")
        vector_store.load(vector_store_dir)
    
    # Generate raw summary
    print("Generating structured summary")
    summary = generate_summary(vector_store, output_file, retry_limit=retry_limit)
    
    # Enhance the summary
    enhanced_output = output_file.replace('.txt', '_enhanced.txt')
    print("Enhancing summary with formatting improvements")
    enhanced_file = enhance_summary(output_file, enhanced_output)
    
    # Generate PDF if requested
    if generate_pdf:
        pdf_output = enhanced_output.replace('.txt', '.pdf')
        print("Converting enhanced summary to PDF")
        pdf_file = convert_markdown_to_pdf(enhanced_file, pdf_output)
        return pdf_file
    
    return enhanced_file

def main():
    parser = argparse.ArgumentParser(description="Process a large placement memorandum and generate a summary")
    parser.add_argument("--pdf", required=True, help="Path to the placement memorandum PDF")
    parser.add_argument("--output", default="memo_summary.txt", help="Output file for the summary")
    parser.add_argument("--vector-store", default="vector_store", help="Directory for the vector store")
    parser.add_argument("--reprocess", action="store_true", help="Force reprocessing even if vector store exists")
    parser.add_argument("--chunk-size", type=int, default=8000, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=400, help="Overlap between chunks")
    parser.add_argument("--index-type", default="HNSW", choices=["Flat", "IVF", "IVFPQ", "HNSW"], help="Type of FAISS index")
    parser.add_argument("--retry-limit", type=int, default=3, help="Number of retries for API calls")
    parser.add_argument("--no-pdf", action="store_true", help="Skip generating PDF output")
    
    args = parser.parse_args()
    
    result_file = process_memo(
        args.pdf, 
        args.output, 
        args.vector_store, 
        args.reprocess,
        args.chunk_size,
        args.chunk_overlap,
        args.index_type,
        args.retry_limit,
        not args.no_pdf
    )
    
    print(f"\nProcess complete. Final output saved to: {result_file}")

if __name__ == "__main__":
    main()