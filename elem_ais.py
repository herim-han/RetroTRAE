from collections import Counter
import re
from rdkit import Chem, RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles, CanonSmiles
import pickle
 
NONE_PHYSICAL_CHARACTERS = ('.', ':',  '-', '=', '#', '(', ')', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '\\', '/', '%')

def encode(smiles, sp_model, with_atomMap=False):
    """ Transforms given SMILES into Atom-in-SMILES (AiS) tokens. By default, it first canonicalizes the input SMILES.
    In order to get AiS tokens with the same order of SMILES, the input SMILES should be provided with atom map number.

    parameters:
        smiles: str, SMILES 
        with_atomMap: if true, it returns AiS with the same order of SMILES.
                      Useful for randomized SMILES, or SMILES augmentation.

    return: 
        str, AiS tokens with white space separated.  
    """
    
#    print(sp_model)
    sp_model = sp_model.replace('model', 'vocab')
    list_token = [ re.split("\s+", line.strip())[0][1:] for line in open(f'{sp_model}').readlines()[4:] ]
    list_token = [ token for token in list_token if ';' in token] #remain only AIS token
#    print('list_token ', list_token, len(list_token))

    smiles_list = smiles.split('.')
    atomInSmiles = []
    for smiles in smiles_list:
        if  with_atomMap:
            mol = MolFromSmiles(smiles)
            if mol is None: return
        else:
            smiles = CanonSmiles(smiles)
#            print('canonical smiles', smiles)
            mol = MolFromSmiles(smiles)
            if mol is None: return

        atomID_sma = {}; atoms = set()
        for idx, atom in enumerate(mol.GetAtoms()):
            atoms.add(atom.GetSmarts())
            atom_symbol = atom.GetSymbol()
            chiral_tag = atom.GetChiralTag().name
            charge = atom.GetFormalCharge()
            nHs = atom.GetTotalNumHs()
    
            if atom.GetIsAromatic(): 
                atom_symbol = atom_symbol.lower()
            if chiral_tag == 'CHI_TETRAHEDRAL_CCW':
                if nHs:
                    symbol = f"[{atom_symbol}@H]"
                else:
                    symbol = f"[{atom_symbol}@]"
            elif chiral_tag == "CHI_TETRAHEDRAL_CW":
                if nHs:
                    symbol = f"[{atom_symbol}@@H]"
                else:
                    symbol = f"[{atom_symbol}@@]"
            else:
                symbol = atom_symbol
                if nHs:
                    symbol += 'H'
                    if nHs>1:
                        symbol += '%d'%nHs
            if charge > 0:
                symbol = symbol.replace('[', '').replace(']', '') 
                symbol = f"[{symbol}+{charge}]" if charge > 1 else  f"[{symbol}+]"
            elif charge < 0:
                symbol = symbol.replace('[', '').replace(']', '') 
                symbol = f"[{symbol}{charge}]" if charge < -1 else  f"[{symbol}-]"
            ring = 'R' if atom.IsInRing() else '!R'
            neighbs = ''.join(sorted([i.GetSymbol() for i in atom.GetNeighbors()]))
            atomID_sma[idx] = f'[{symbol};{ring};{neighbs}]'

        ais = []
        idx=0
        for token in smiles_tokenizer(smiles):
            if any ([ token.startswith(NONE_PHYSICAL_CHARACTER) for NONE_PHYSICAL_CHARACTER in NONE_PHYSICAL_CHARACTERS ]):
                symbol = token
            else:
                if atomID_sma[idx] in list_token:
                    symbol = atomID_sma[idx]
                else:
                    symbol = token 
                idx+=1
            ais.append(symbol)
        atomInSmiles.append(' '.join(ais))

    return ' . '.join(atomInSmiles)

def org_encode(smiles, with_atomMap=False):
    """ Transforms given SMILES into Atom-in-SMILES (AiS) tokens. By default, it first canonicalizes the input SMILES.
    In order to get AiS tokens with the same order of SMILES, the input SMILES should be provided with atom map number.

    parameters:
        smiles: str, SMILES 
        with_atomMap: if true, it returns AiS with the same order of SMILES.
                      Useful for randomized SMILES, or SMILES augmentation.

    return: 
        str, AiS tokens with white space separated.  
    """
    smiles_list = smiles.split('.')
    atomInSmiles = []
    for smiles in smiles_list:
        if  with_atomMap:
            mol = MolFromSmiles(smiles)
            if mol is None: return 
        else:
            tmp = MolFromSmiles(CanonSmiles(smiles))
            if tmp is None: return 
            for atom in tmp.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx())
            smiles = MolToSmiles(tmp)
            mol = MolFromSmiles(smiles)
        atomID_sma = {}; atoms = set()
        for atom in mol.GetAtoms():
            atoms.add(atom.GetSmarts())
            try: atomId = atom.GetPropsAsDict()['molAtomMapNumber']
            except: atomId = 0
            atom_symbol = atom.GetSymbol()
            chiral_tag = atom.GetChiralTag().name
            charge = atom.GetFormalCharge()
            nHs = atom.GetTotalNumHs()

            if atom.GetIsAromatic(): 
                atom_symbol = atom_symbol.lower()
            if chiral_tag == 'CHI_TETRAHEDRAL_CCW':
                if nHs:
                    symbol = f"[{atom_symbol}@H]"
                else:
                    symbol = f"[{atom_symbol}@]"
            elif chiral_tag == "CHI_TETRAHEDRAL_CW":
                if nHs:
                    symbol = f"[{atom_symbol}@@H]"
                else:
                    symbol = f"[{atom_symbol}@@]"
            else:
                symbol = atom_symbol
                if nHs:
                    symbol += 'H'
                    if nHs>1:
                        symbol += '%d'%nHs                           
            symbol = symbol.replace('[','').replace(']', '') 
            if charge > 0:
                symbol = f"[{symbol}+{charge}]" if charge > 1 else  f"[{symbol}+]"
            elif charge < 0:
                symbol = f"[{symbol}{charge}]" if charge < -1 else  f"[{symbol}-]"
            ring = 'R' if atom.IsInRing() else '!R'
#            for i in atom.GetNeighbors():
#                print(i.GetSymbol())
#                print(''.join(sorted([i.GetSymbol()])))
            neighbs = ''.join(sorted([i.GetSymbol() for i in atom.GetNeighbors()]))
            atomID_sma[atomId] = f'[{symbol};{ring};{neighbs}]'

        ais = []
        for token in smiles_tokenizer(smiles):
            if token in NONE_PHYSICAL_CHARACTERS:
                symbol = token
            else:
                try: atom, atomId = token[1:-1].split(':')
                except: atom, atomId = token, 0
                symbol = atomID_sma[int(atomId)]
            ais.append(symbol)

        atomInSmiles.append(' '.join(ais))

    return ' . '.join(atomInSmiles)

def decode(atomInSmiles):
    """ Converts Atom-in-SMILES tokens  back to SMILES string.
    Note: The Atom-in-SMILES tokens should be white space separated.
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    smarts = ''
    for new_token in atomInSmiles.split():
        try:
            token = regex.findall(new_token)[0]
        except:
            token = new_token
        if '[[' in token:
            smarts += token[1:]
        elif ';' not in token:
            smarts+=token
        else:
            if '[' in token:
                sym = token[1:-1].split(';')
                if sym[0] =='nH':
                    smarts += '[nH]'
                elif "H" in sym[0]:
                    smarts += regex.findall(sym[0])[0]
                else:
                    smarts += sym[0]
            else:
                smarts += token
    return smarts


def similarity(ais1, ais2):
    """ Tanimoto coefficient of two AiS tokens. Here AiS tokens are treated as fingperprint 
        which means non-physical characters will be removed.
    """
    ais1_atoms = [ i for i in ais1.split() if i not in NONE_PHYSICAL_CHARACTERS]
    ais2_atoms = [ i for i in ais2.split() if i not in NONE_PHYSICAL_CHARACTERS]
    a, b = Counter(ais1_atoms), Counter(ais2_atoms)
    
    a_or_b = set(a).union(set(b))
    a_and_b = set(a).intersection(set(b))
    
    sum_intersection = 0
    for key in a_and_b:
        i, j = a[key], b[key]
        sum_intersection += max(i,j) - abs(i-j)

    sum_union = 0
    for key in a_or_b:
        i,j = a[key], b[key]
        sum_union += max(i, j)
        
    return sum_intersection/sum_union


def smiles_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens #' '.join(tokens)

if __name__ == '__main__':
    #smiles = 'COCCNC(=O)Cn1c2ccccc2c2c1[C@H]1N(C)c3ccccc3C(=O)N1CC2'
    #smiles = 'Oc1ccc(c(c1)O)C(=O)N(C1Cc2c(C1)cccc2)c1ccc(cc1)C(=O)N(C)C'
    smiles = 'COc1ccc(-c2ccccc2S(N)=Nc2ccccc2O)cc1'
    print(smiles)
    ais = encode(smiles, './../data/sp/ais_vocab_200+312.model')
    print(ais)
    smi = decode(ais)
    print(smi)
    exit(-1)
    for smiles in [line.strip() for line in open('zinc_10', 'r').readlines()]:
        print(smiles)
        ais = encode(smiles, 'C')
        #ais = org_encode(smiles, False)
        print(ais)
        smi = decode(ais)
        print(smi)
    #print(smi)
