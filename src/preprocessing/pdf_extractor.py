import re
import pymupdf
import pandas as pd
from typing import Dict, Optional, List
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFMedicalExtractor:
    """
    - Text extraction from PDF
    - Pattern matching for clinical fields
    - Protocol normalization
    - Monitoring table parsing
    """
    
    # Protocol mapping for normalization
    PROTOCOL_MAPPING = {
        'flex antago': 'flexible antagonist',
        'flexible antago': 'flexible antagonist',
        'flex anta': 'flexible antagonist',
        'fix antag': 'fixed antagonist',
        'fixed antag': 'fixed antagonist',
        'fix anta': 'fixed antagonist',
        'fixed anta': 'fixed antagonist',
        'agonist': 'agonist'
    }
    
    # Response mapping
    RESPONSE_MAPPING = {
        'low-response': 'low',
        'optimal-response': 'optimal',
        'high-response': 'high',
        'low response': 'low',
        'optimal response': 'optimal',
        'high response': 'high'
    }
    
    def __init__(self):
        self.existing_ids = set()
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        raises: FileNotFoundError: If PDF file doesn't exist
                Exception: If PDF cannot be read
        """
        try:
            doc = pymupdf.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
                
            doc.close()
            logger.info(f"Successfully extracted text from {pdf_path}")
            return text
            
        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise
    
    def generate_patient_id(self, name: str, existing_csv_path: Optional[str] = None) -> str:
        #Generate de-identified patient ID from name; 25XXX where XXX is the row number in CSV.

        # load existing CSV to get row count
        if existing_csv_path and Path(existing_csv_path).exists():
            df = pd.read_csv(existing_csv_path)
            next_row = len(df) + 1  # next row number
        else:
            next_row = 1  # first patient
        
        patient_id = f"25{next_row:02d}"  # format 25XXX
        
        logger.info(f"Generated patient ID {patient_id} for {name} (row {next_row})")
        return patient_id
    
    def normalize_protocol(self, protocol_text: str) -> str:
        #Normalize protocol names to standard values

        if not isinstance(protocol_text, str) or not protocol_text.strip():
            logger.warning(f"Invalid protocol value: {protocol_text}")
            return protocol_text

        protocol_lower = protocol_text.lower().strip()

        # substring matching
        for key, value in self.PROTOCOL_MAPPING.items():
            if key in protocol_lower:
                return value

        # if no match found
        logger.warning(f"Unknown protocol: '{protocol_text}'")
        return protocol_text

    
    def parse_patient_data(self, text: str, existing_csv_path: Optional[str] = None) -> Dict:
        """
        Parse clinical data from extracted PDF text using regex patterns :
            - Patient name (for ID generation)
            - Cycle number
            - Age (calculated from birth date)
            - Protocol
            - AMH level
            - AFC (Antral Follicle Count)
            - Number of follicles
            - E2 on day 5
            - Patient response
        
        Returns a dict with extracted patient data
        """
        data = {}
        
        # Extract patient name
        name_match = re.search(r'Name\s*:\s*([A-Za-z\s]+)', text, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
            data['patient_id'] = self.generate_patient_id(name, existing_csv_path)
        else:
            logger.warning("Patient name not found")
            data['patient_id'] = None
        
        #extract cycle number
        cycle_match = re.search(r'Cycle number\s*:\s*(\d+)', text, re.IGNORECASE)
        if cycle_match:
            data['cycle_number'] = int(cycle_match.group(1))
        else:
            logger.warning("Cycle number not found")
            data['cycle_number'] = None
        
        #extract birth date and calculate age
        birth_match = re.search(r'Birth date:\s*(\d{2})/(\d{2})/(\d{2})', text)
        if birth_match:
            day, month, year = birth_match.groups()
            # Assume 20XX for years < 25, 19XX for years >= 25
            full_year = int(f"20{year}") if int(year) < 25 else int(f"19{year}")
            data['Age'] = 2025 - full_year

        else:
            logger.warning("Birth date not found")
            data['Age'] = None
        
        # Extract protocol
        protocol_match = re.search(r'Protocol\s*:\s*([^\n]+)', text, re.IGNORECASE)
        if protocol_match:
            protocol = protocol_match.group(1).strip()
            data['Protocol'] = self.normalize_protocol(protocol)
        else:
            logger.warning("Protocol not found")
            data['Protocol'] = None
        
        # Extract AMH
        amh_match = re.search(r'AMH\s*:\s*([\d.]+)', text, re.IGNORECASE)
        if amh_match:
            data['AMH'] = float(amh_match.group(1))
        else:
            logger.warning("AMH not found")
            data['AMH'] = None
        
        # Extract AFC
        afc_match = re.search(r'AFC\s*:\s*(\d+)', text, re.IGNORECASE)
        if afc_match:
            data['AFC'] = int(afc_match.group(1))
        else:
            logger.warning("AFC not found, will leave empty")
            data['AFC'] = None
        
        # Extract number of follicles
        follicles_match = re.search(r'Number [Oo]f follicles\s*=\s*(\d+)', text)
        if follicles_match:
            data['n_Follicles'] = int(follicles_match.group(1))
        else:
            logger.warning("Number of follicles not found")
            data['n_Follicles'] = None
        
        # Extract E2 on day 5 from monitoring table
        e2_match = re.search(r'J\s*5.*?(\d+(?:\.\d+)?)\s+[\d.]+\s+\d+', text, re.DOTALL) # we ll look for J5 row with E2 value
        if e2_match:
            data['E2_day5'] = float(e2_match.group(1))
        else:
            logger.warning("E2 day 5 not found in monitoring table")
            data['E2_day5'] = None
        
        # Extract patient response
        response_match = re.search(r'patient has an?\s+([a-z-]+(?:\s+)?response)', text, re.IGNORECASE)
        if response_match:
            response = response_match.group(1).strip()
            data['Patient Response'] = self.RESPONSE_MAPPING.get(response.lower(), response)
        else:
            logger.warning("Patient response not found")
            data['Patient Response'] = None
        
        return data
    
    def extract_from_pdf(self, pdf_path: str, existing_csv_path: Optional[str] = None) -> Dict:
        # Complete extraction pipeline: PDF -> structured data
        logger.info(f"Starting extraction from {pdf_path}")
        
        # 1.Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # 2.Parse data
        data = self.parse_patient_data(text, existing_csv_path)
        
        #3. Validate data completeness
        missing_fields = [k for k, v in data.items() if v is None]
        if missing_fields:
            logger.warning(f"Missing fields: {', '.join(missing_fields)}")
        
        logger.info("Extraction completed successfully")
        return data
    
    def add_to_csv(self, pdf_path: str, csv_path: str, output_path: Optional[str] = None):
        # Extract data
        new_data = self.extract_from_pdf(pdf_path, csv_path)
        
        # Load existing CSV
        df = pd.read_csv(csv_path)
        
        # create new row
        new_row = pd.DataFrame([new_data])
        
        #append to dataframe
        df_updated= pd.concat([df, new_row], ignore_index=True)
        


        output = output_path if output_path else csv_path
        df_updated.to_csv(output, index=False)
        
        logger.info(f"Added new patient record to {output}")
        logger.info(f"New dataset size: {len(df_updated)} patients")


def main():
    extractor = PDFMedicalExtractor()
    
    # file paths
    import os
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Raw and processed data folders
    RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw")
    PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed")

    pdf_path = os.path.join(RAW_DATA, "sample.pdf")
    csv_path = os.path.join(RAW_DATA, "patients.csv")
    output_path = os.path.join(PROCESSED_DATA, "patients_updated.csv")

    # Extract and add to CSV
    try:
        extractor.add_to_csv(pdf_path, csv_path, output_path)
        print("\n✅ PDF extraction completed successfully!")
        print(f"📊 Updated dataset saved to: {output_path}")
        
        # Display extracted data
        df = pd.read_csv(output_path)
        print("\n📋 Newly added patient:")
        print(df.tail(1).to_string(index=False))
        
    except Exception as e:
        print(f"\n❌ Error during extraction: {str(e)}")
        raise


if __name__ == "__main__":
    main()