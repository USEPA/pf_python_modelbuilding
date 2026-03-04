


from indigo import Indigo
from indigo.renderer import IndigoRenderer
from indigo.inchi import IndigoInchi

from typing import Optional, Dict
import base64
import pandas as pd

class IndigoUtils :
    
    def __init__(self):
        self.indigo = Indigo()
        self.renderer = IndigoRenderer(self.indigo)
        self.indigo_inchi = IndigoInchi(self.indigo)
        
        self.indigo.setOption("render-output-format", "png")

    
    def inchi_key_from_smiles(self, smiles: str, short_key: bool = False) -> Optional[str]:
        """
        Convert SMILES to an InChIKey.
        Set short_key=True to return only the first 14 characters (connectivity block).
        """
        if smiles is None or str(smiles).strip() == "":
            return None
        try:
            mol = self.indigo.loadMolecule(str(smiles))
            inchi_str = self.indigo_inchi.getInchi(mol)
            ik = self.indigo_inchi.getInchiKey(inchi_str)
            if ik is None:
                return None
            # Return first block (14 chars); using split ensures robustness even if format changes.
            return ik.split("-")[0] if short_key else ik
        except Exception as e:
            print(f"Error getting InChIKey for SMILES '{smiles}': {e}")
            return None

    def smiles_to_inchikey_dict(
        self,
        df: pd.DataFrame,
        smiles_col: str = "canon_qsar_smiles",
        short_key: bool = False,
        reverse_lookup = False, 
    ) -> Dict[str, Optional[str]]:
        """
        Build a dict mapping unique SMILES to InChIKey (or first 14 chars if short_key=True).
        """
        unique_smiles = pd.Series(df[smiles_col]).dropna().unique()
        smiles_to_key: Dict[str, Optional[str]] = {}

        for smi in unique_smiles:
            try:
                ik = self.inchi_key_from_smiles(smi, short_key=short_key)
                
                if reverse_lookup:
                    smiles_to_key[ik] = smi
                else:
                    smiles_to_key[smi] = ik
                    
            except Exception as e:
                print(f"Failed SMILES '{smi}': {e}")
                smiles_to_key[smi] = None

        return smiles_to_key
    
        
    def smiles_png_b64_indigo(self, smiles_string, width=240, height=180,trim=False):
        '''
        TODO: move to utility class
        :param smiles_string:
        '''
        try:
            # print(width, height)            
            mol = self.indigo.loadMolecule(smiles_string)
            self.indigo.setOption("render-output-format", "png") 
            self.indigo.setOption("render-image-width", width)
            self.indigo.setOption("render-image-height", height)
            img_bytes = self.renderer.renderToBuffer(mol)
                        
            if trim:
                import io
                from PIL import Image, ImageChops
                image = Image.open(io.BytesIO(img_bytes))
    
                # 2. Define the "blank" background (assuming top-left pixel is the background color)
                # If your background is always white, use: bg = Image.new(image.mode, image.size, (255, 255, 255))
                bg = Image.new(image.mode, image.size, image.getpixel((0,0)))
                
                # 3. Find the difference to get the bounding box of non-blank content
                diff = ImageChops.difference(image, bg)
                bbox = diff.getbbox() # Returns (left, upper, right, lower)
                
                if bbox:
                    # 4. To trim ONLY top and bottom, keep the original width (0 to image.size[0])
                    # and use the calculated upper/lower bounds
                    left, upper, right, lower = bbox
                    trimmed_image = image.crop((0, upper, image.size[0], lower))
                    
                else:
                    trimmed_image = image
                 
                out_buffer = io.BytesIO()
                trimmed_image.save(out_buffer, format="PNG")
                img_bytes = out_buffer.getvalue()
            
            
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            return base64_string
    
        except Exception as e:
            print(f"An error occurred while loading the molecule: {e}")
            return None