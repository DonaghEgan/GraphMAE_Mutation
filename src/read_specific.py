import os
import re
import torch
from torch_geometric.data import download_url, extract_zip, extract_gz
import numpy as np
from torch_geometric.data import Data
import csv
from . import download_study
import zipfile 
from typing import Optional, List, Dict  

def read_reactome_new(gene_index: Dict[str, int], 
                      folder: str = 'temp/', 
                      url: str = 'https://reactome.org/download/tools/ReatomeFIs/FIsInGene_122921_with_annotations.txt.zip') -> torch.Tensor:
    """
    Function to read Reactome file and extract gene regulatory and protein-protein interaction networks.

    :param tokens: A list of gene symbols to filter the network. If None, all genes are included.
    :param folder: Directory where the file will be downloaded and extracted.
    :return: A dictionary with:
        - tokens: List of gene symbols.
    """
    # test url
    if url is None or not isinstance(url, str):
        raise ValueError('url must be provided, and in string format')

    # download
    try:
        path = download_url(url, folder) 
    except Exception as e:
        raise RuntimeError(f"Failed to download or extract file: {e}")
    
    # if it is a zip file extract
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(folder)
    
    # takes first filename found
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            extracted_path = os.path.join(folder, filename)
            break
    else:
        raise FileNotFoundError("No .txt file found in the folder.")

    # Read file and store parsed lines (exlcuding header)
    interaction_pairs = []

    with open(extracted_path, 'r') as fo:
        for i, line in enumerate(fo):
            if i == 0:
                continue
            parts = line.rstrip('\n').split('\t')
            # remove low scoring interactions
            if float(parts[4]) >= 0.9:
                interaction_pairs.append([parts[0], parts[1]])

    # Initialize adjacency matrix
    adj_matrix = torch.zeros((len(gene_index), len(gene_index)), dtype = torch.float32)
    for gene_a, gene_b in interaction_pairs:
        if gene_a in gene_index and gene_b in gene_index:
             idx_a = gene_index[gene_a]    
             idx_b = gene_index[gene_b]
             adj_matrix[idx_a, idx_b] = 1
             adj_matrix[idx_b, idx_a] = 1

    # Test known interactions.
    known_interactions = [('AXL', 'EGFR'), ('TP53', 'BRCA1'), ('KRAS', 'PTPN11')]

    for gene_a, gene_b in known_interactions:
        if gene_a in gene_index and gene_b in gene_index:
             idx_a = gene_index[gene_a]    
             idx_b = gene_index[gene_b]
             assert adj_matrix[idx_a, idx_b] == 1, f"Interaction between {gene_a} and {gene_b} not found in adjacency matrix."

    return adj_matrix


def download_pathway_sets(folder = 'temp/'):
    url = 'https://reactome.org/download/current/ReactomePathways.gmt.zip'
    path = download_url(url, folder)
    extract_zip(path, folder)
    path = path[0:len(path)-4]
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    # read what genes belong 
    # First column is pathway name, then code, then gene set.
    gene_set_dict = dict()
    # go through limelist object in a for loop and split.
    for i in lineList:
        div = re.split('\t', i)
        descrip = div[0]
        code = div[1]
        set = div[2:]
        gene_set_dict[code] = {'description':descrip, 'set': set}
    return(gene_set_dict)


def download_read_gos(folder = 'temp/'):
    ontology_description = 'http://current.geneontology.org/ontology/go-basic.obo' 
    path = download_url(ontology_description, folder)
    lineList = [line.rstrip('\n') for line in open(path)]
    # we'll return three outputs, 
    # 1 tree of ontologies.
    onto_tree = dict()
    # 2 dict of ontologies.
    onto_dict = dict()
    # 3 ontology by ctype
    onto_type = dict() # for simplicity, we could use the previous one. 
    j = 0
    # Do a while loop. 
    while j < len(lineList):
        # see if we find a line that starts with id then store ontology id.
        if lineList[j] == '[Term]':
            j += 1
            onto_id = re.split(' ',lineList[j])[1]
            j +=1
            onto_desc = re.split(':', lineList[j])[1]
            j += 1
            onto_namespace = re.split(':', lineList[j])[1][1:]
            onto_sub = list()
            while lineList[j]!= '':
                 j += 1
                 div = re.split(' ',  lineList[j])
                 if div[0] == 'is_a:':
                     onto_sub.append(div[1])
            # add elements to onto trees
            onto_tree[onto_id] = onto_sub
            # add elements to onto_dict
            onto_dict[onto_id] = {'name':onto_desc, 'name space':onto_namespace, 'part of':onto_sub}
            # add to onto_type
            if list(onto_type.keys()).count(onto_namespace)>0:
                onto_type[onto_namespace].append(onto_id)
            else:
                onto_type[onto_namespace] = [onto_id]
        else:
            j+= 1
    return(onto_type, onto_dict, onto_tree)



def download_gene_gos(folder = 'temp/'):
    ontology_annot = 'http://geneontology.org/gene-associations/goa_human.gaf.gz'
    path = download_url(ontology_annot, folder)
    extract_gz(path, folder)
    path = path[0:len(path)-3]
    fo = open(path)
    lineList = [line.rstrip('\n') for line in fo]
    fo.close()
    gene_onto = dict()
    for i in lineList:
        # these file has multiple lines starting with ! to discard
        if i[0] != '!':
            div = re.split('\t', i) # Tab separate the entries.
            gene_id  = div[2] # get SYMBOL
            onto_i = div[4] # Ontology code. 
            if list(gene_onto.keys()).count(gene_id) >0 :
                if gene_onto[gene_id].count(onto_i)==0:
                    gene_onto[gene_id].append(onto_i)
            else:
                gene_onto[gene_id] = [onto_i]
    return(gene_onto)

def download_onco_tree(folder = 'temp/'):
    onco_url = 'http://oncotree.mskcc.org/api/tumor_types.txt'
    path = download_url(onco_url, folder)
    lineList = [line.rstrip('\n') for line in open(path)]
    onco_tree = dict()
    onco_dict = dict()
    # do a for loop and store in dictioanry with two fields. 
    # field 1, directed graph. 
    # field 2 the main name. 
    for i in range(1, len(lineList)):
        div = re.split('\t', lineList[i])
        # first field is the level 1. get the code from within brackets.
        onco_codes = list()
        j =0
        while div[j]!= '':
            splt = re.split(r'[\(\)]', div[j])
            # check if term is in dictionary already. 
            if list(onco_dict.keys()).count(splt[1]) ==0:
                onco_dict[splt[1]] = splt[0]
            onco_codes.append(splt[1])
            j += 1
        # add to onco_tree dictionary.
        onco_tree[onco_codes[len(onco_codes)-1]] = onco_codes
    return(onco_tree, onco_dict)

def download_read_msk_2017(folder = 'temp/', top = 15):
    path, sources, urls = download_study(name = 'msk_pan_2017', folder = folder)
    ret_dict = read_folders.read_cbioportal_folder(path)
    # read the onco_tree and onco_dict.
    onco_tree, onco_dict = download_onco_tree(folder)
    # we need to put the outputs in the following formats.
    sample_proc = msk_2017_sample_info(ret_dict['sample']['sample_dict'])
    # and process then the patient data.
    pat_proc = msk_2017_patient_sample(sample_dict,patient_dict)
     
def msk_2017_sample_info(sample_dict, top = 15):
    # Finally, let's go through the sample and clinical datasets.
    # From the sample dictionary we are interested in a couple of things.
    # First: What patient it came from? We will use this for survival and clinical variables. Key: 'Patient Identifier'
    pats = list()
    # Second: What oncotree code is the sample from? Key; 'Oncotree Code'
    onco_codes = dict()
    # Third: Is it primary or metastatic sample. Key: 'Sample Type'
    sample_type = dict()
    # Fourth: Primary tumour Site. Key: 'Primary Tumor Site'.
    tumor_sites = dict()
    # Fifth: Metastatic site. Key: 'Metastatic Site'.
    metas_sites = dict()
    # This is a bit of an odd procedure. We will first create a list
    for i in sample_dict:
        p = sample_dict[i]['Patient Identifier']
        if pats.count(p) == 0:
            pats.append(p)
        # add unique onco codes
        onc = sample_dict[i]['Oncotree Code']
        onc_keys = list(onco_codes.keys())
        if onc_keys.count(onc) == 0:
            onco_codes[onc] = 1
        else:
            onco_codes[onc] += 1
        s = sample_dict[i]['Sample Type']
        samp_keys = list(sample_type.keys())
        if samp_keys.count(s) == 0:
            sample_type[s] = 1
        else:
            sample_type[s] += 1
        ts = sample_dict[i]['Primary Tumor Site']
        tum_keys = list(tumor_sites.keys())
        if tum_keys.count(ts) == 0:
            tumor_sites[ts] = 1
        else:
            tumor_sites[ts] += 1
        ms = sample_dict[i]['Metastatic Site']
        metas_keys = list(metas_sites.keys())
        if metas_keys.count(ms) == 0:
            metas_sites[ms] = 1
        else:
            metas_sites[ms] += 1
    # Now we wish to encode each into a nice matrix. 
    # A couple of situations have risen.
    # we need to encode everything in nummerical form.
    # We will use the onco trees to encode into nummerical matrixes.
    summary = {'patients': pats, 'onco_codes': onco_codes, 'sample_types': sample_type, 'tumor_sites': tumor_sites, 'metas_sites': metas_sites}
    n = len(sample_dict.keys())
    metas_site_bool = np.zeros((n,1))# Matrix for whether there was a metastasic site associated. 
    sample_type_bool = np.zeros((n,2)) # Matrix to indiciate if a sample is from a metastasis or not. 
    # we will select the top common sites (to avoid cancer sites with only 1 or 2 samples for example.)
    metas_site = np.zeros((n, top))
    top_meta = np.argsort(-1*np.array(list(metas_sites.values())))[0:top]
    top_meta_list = list()
    meta_keys = list(metas_sites.keys())
    for i in range(top_meta.shape[0]):
        top_meta_list.append(meta_keys[top_meta[i]])
    # same with tumour type. 
    tumour_site = np.zeros((n, top)) 
    top_tumor = np.argsort(-1*np.array(list(tumor_sites.values())))[0:top]
    top_tumor_list = list()
    tum_keys = list(tumor_sites.keys())
    for i in range(top_tumor.shape[0]):
        top_tumor_list.append(tum_keys[top_tumor[i]])
    # we will download the onco tree dictionary and make a onco encoding? # yeap that sounds correct I supose. 
    oncotree = np.zeros((n, 6)) # there's a maximum of 6 levels for this things. we will take the integers from the oncotree associated.
    # download and read the oncotree.
    onco_tree, onco_dict = download_onco_tree()
    # make a dictionary where we encode each onco code into an integer.
    onco_keys = list(onco_dict.keys())
    j = 0
    for i in sample_dict:
        # Fill in meta_site_pool
        if sample_dict[i]['Metastatic Site'] != 'Not Applicable':
            metas_site_bool[j,0] = 1
        # Next sample type
        if sample_dict[i]['Sample Type'] == 'Primary':
            sample_type_bool[j,0] = 1
        else:
            sample_type_bool[j,1] = 1
        # Write the top metastatic sites. 
        if top_meta_list.count(sample_dict[i]['Metastatic Site'])>0 :
            idx = top_meta_list.index(sample_dict[i]['Metastatic Site'])
            metas_site[j,idx] = 1
        if top_tumor_list.count(sample_dict[i]['Primary Tumor Site'])>0 :
            idx = top_tumor_list.index(sample_dict[i]['Primary Tumor Site'])
            tumour_site[j,idx] = 1
        # Encode the oncotree sitch!
        onco_code = sample_dict[i]['Oncotree Code']
        if list(onco_tree.keys()).count(onco_code)>0:
            onco_situation = onco_tree[onco_code]
            for k in range(len(onco_situation)):
                no = onco_keys.index(onco_situation[k])
                oncotree[j,k] = no 
        j += 1
    output = dict()
    output['summary'] = summary
    output['metas_site_bool']  = metas_site_bool
    output['sample_type_bool'] = sample_type_bool
    output['metas_site'] = metas_site
    output['tumour_site'] = tumour_site
    output['oncotree']  = oncotree
    return(output)



def msk_2017_patient_sample(sample_dict,patient_dict):
    # get keys in sample_dict and put the survival and age into numpy arrays.
    sample_keys = list(sample_dict.keys())
    # get number of obvs. 
    n = len(sample_keys)
    # Pre-allocate memory for Overall Survival, Age, Gender, Smoking
    smoking = np.zeros((n,2)) # Column 1 Never, Column 2 Prev/Curr Smoker, (other value is unknown)
    osurv = np.zeros((n,2)) # First column is overall survival in days 
    gender = np.zeros((n,1)) # We will for simplicity just use one gender as baseline. 
    status = np.zeros((n,1))
    # Do a for loop.
    p_inf= dict()
    for i in patient_dict:
        info = patient_dict[i]
        # make a dictionary with the data 
        pat_info = dict()
        pat_info['gender'] = 1*(info[0]=='Female')
        pat_info['status'] = 1*(info[1]=='DECEASED')
        if info[2] == 'Never':
            pat_info['smoking'] = 1
        elif info[2] == 'Prev/Curr Smoker':
            pat_info['smoking'] = 2
        else:
            pat_info['smoking'] = 0
        if info[3] != '':
            pat_info['survival_time'] = float(info[3])
        else:
            pat_info['survival_time'] = 0
        if info[4] == '1:DECEASED':
            pat_info['censored'] =1
        else:
            pat_info['censored'] = 0.
        p_inf[i] = pat_info
    j = 0
    for i in sample_dict:
        # get patient id. 
        pat_id = sample_dict[i]['Patient Identifier']
        # get info.
        info = p_inf[pat_id]
        # gender
        gender[j,0] = info['gender']
        # smoking
        if info['smoking'] >0 :
            smoking[j,info['smoking']-1] = 1.
        # survival
        osurv[j,0] = info['survival_time']
        osurv[j,1] = info['censored']
        # status
        status[j,0] = info['status']
        j += 1
    out = dict()
    out['pat_info'] = p_inf
    out['status'] = status
    out['gender'] = gender
    out['smoking'] = smoking
    out['osurv'] = osurv
    return(out)

