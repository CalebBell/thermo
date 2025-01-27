'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, 2018, 2019, 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''
from chemicals.elements import simple_formula_parser

__all__ = ['str_group_assignment_to_dict', 'group_assignment_to_str',
           'smarts_fragment_priority', 'smarts_fragment', 'priority_from_atoms',
           'SINGLE_BOND', 'DOUBLE_BOND', 'TRIPLE_BOND', 'AROMATIC_BOND',
           'BaseGroupContribution']

SINGLE_BOND = 'single'
DOUBLE_BOND = 'double'
TRIPLE_BOND = 'triple '
AROMATIC_BOND = 'aromatic'

def priority_from_atoms(atoms, bonds=None):
    priority = 0

    if 'H' in atoms:
        priority += atoms['H']

    if 'C' in atoms:
        priority += atoms['C']*100

    if 'O' in atoms:
        priority += atoms['O']*150
    if 'N' in atoms:
        priority += atoms['N']*175
    if 'Cl' in atoms:
        priority += atoms['Cl']*300
    if 'F' in atoms:
        priority += atoms['F']*400
    if 'Si' in atoms:
        priority += atoms['Si']*200
    if 'S' in atoms:
        priority += atoms['S']*250

    if bonds is not None:
        priority += bonds.get(SINGLE_BOND, 0)*2
        priority += bonds.get(DOUBLE_BOND, 0)*10
        priority += bonds.get(TRIPLE_BOND, 0)*100
        priority += bonds.get(AROMATIC_BOND, 0)*1000
    return priority



rdkit_missing = 'RDKit is not installed; it is required to use this functionality'

loaded_rdkit = False
Chem, Descriptors, AllChem, rdMolDescriptors = None, None, None, None
def load_rdkit_modules():
    global loaded_rdkit, Chem, Descriptors, AllChem, rdMolDescriptors, combinations
    if loaded_rdkit:
        return
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
        loaded_rdkit = True
        from itertools import combinations
    except:
        if not loaded_rdkit: # pragma: no cover
            raise Exception(rdkit_missing)

class BaseGroupContribution:
    __slots__ = ('group', 'group_id', 'smarts', 'smart_rdkit', 
                 'hydrogen_from_smarts', 'priority', 'atoms', 'bonds')
    
    def __init__(self, group, smarts=None, priority=None, atoms=None, 
                 bonds=None, hydrogen_from_smarts=False, group_id=None):
        self.group = group
        self.smarts = smarts
        self.priority = priority
        self.atoms = atoms
        self.bonds = bonds
        self.hydrogen_from_smarts = hydrogen_from_smarts
        self.smart_rdkit = None
        self.group_id = group_id



def group_assignment_to_str(counts, pair_separator=',', key_val_separator=':'):
    r'''Take a group contribution dictionary, and turn it into a string.
    The string is usually more memory efficient.

    Parameters
    ----------
    counts : dict
        Dictionaty of integer counts of the found groups only, indexed by
        the `group` [-]
    pair_separator : str, optional
        The separator between pairs of group assignments, [-]
    key_val_separator : str, optional
        The separator between the group, and the amount of that group, [-]

    Returns
    -------
    counts : str
        String of counts of the found groups only [-]

    Notes
    -----

    Examples
    --------
    >>> group_assignment_to_str({1: 5, 4: 1, 9: 5, 10: 1, 81: 1})
    '1:5,4:1,9:5,10:1,81:1'
    '''
    elements = []
    for k, v in counts.items():
        elements.append(f"{k}{key_val_separator}{v}")
    return pair_separator.join(elements)

def str_group_assignment_to_dict(counts, pair_separator=',', key_val_separator=':'):
    r'''Take a group contribution string, and turn it into a dictionary.

    Parameters
    ----------
    counts : str
        String of counts of the found groups only [-]
    pair_separator : str, optional
        The separator between pairs of group assignments, [-]
    key_val_separator : str, optional
        The separator between the group, and the amount of that group, [-]

    Returns
    -------
    counts : dict
        Dictionaty of integer counts of the found groups only, indexed by
        the `group` [-]

    Notes
    -----

    Examples
    --------
    >>> str_group_assignment_to_dict('1:5,4:1,9:5,10:1,81:1')
    {1: 5, 4: 1, 9: 5, 10: 1, 81: 1}
    >>> str_group_assignment_to_dict('')
    {}
    '''
    if not counts:
        return {}
    groups = {}
    for group in counts.split(pair_separator):
        k, v = group.split(key_val_separator)
        groups[int(k)] = int(v)
    return groups




def smarts_fragment_priority(catalog, rdkitmol=None, smi=None):
    r'''Fragments a molecule into a set of unique groups and counts as
    specified by the `catalog`, which is a list of objects containing
    the attributes `smarts`, `group`, and `priority`.

    The molecule can either be an rdkit
    molecule object, or a smiles string which will be parsed by rdkit.
    Returns a dictionary of groups and their counts according to the
    priorities of the catalog provided, as well as some other information

    Parameters
    ----------
    catalog : list
        List of objects, [-]
    rdkitmol : mol, optional
        Molecule as rdkit object, [-]
    smi : str, optional
        Smiles string representing a chemical, [-]

    Returns
    -------
    counts : dict
        Dictionaty of integer counts of the found groups only, indexed by
        the `group` [-]
    group_assignments : dict[`group`: [(matched_atoms_0,), (matched_atoms_0,)]]
        A dictionary of which atoms were included in each of the groups identified, [-]
    matched_atoms : set
        A set of all of the atoms which were matched, [-]
    success : bool
        Whether or not molecule was fully fragmented, [-]
    status : str
        A string holding an explanation of why the molecule failed to be
        fragmented, if it fails; 'OK' if it suceeds.

    Notes
    -----
    Raises an exception if rdkit is not installed, or `smi` or `rdkitmol` is
    not defined.

    Examples
    --------
    '''
    if not loaded_rdkit:
        load_rdkit_modules()
    if rdkitmol is None and smi is None:
        raise Exception('Either an rdkit mol or a smiles string is required')
    if type(rdkitmol) is str and smi is None:
        # swap for convinience
        rdkitmol, smi = smi, rdkitmol
    if smi is not None:
        rdkitmol = Chem.MolFromSmiles(smi)
        if rdkitmol is None:
            status = 'Failed to construct mol'
            success = False
            return {}, success, status
    

    # Remove this
    catalog = [i for i in catalog if i.priority is not None]

    rdkitmol_Hs = Chem.AddHs(rdkitmol)
    # H_count = rdkitmol_Hs.GetNumAtoms() - rdkitmol.GetNumAtoms()
    atoms = simple_formula_parser(rdMolDescriptors.CalcMolFormula(rdkitmol))
    H_count = atoms.get('H', 0)

    H_counts_by_idx = {}
    all_atom_idxs = set()
    for at in rdkitmol.GetAtoms():
        at_idx = at.GetIdx()
        H_counts_by_idx[at_idx] = at.GetTotalNumHs(includeNeighbors=True)
        all_atom_idxs.add(at_idx)

    atom_count = len(all_atom_idxs)
    status = 'OK'
    success = True

    counts = {}
    all_matches = {}
    for obj in catalog:
        patt = obj.smart_rdkit
        if patt is None:
            smart = obj.smarts
            if isinstance(smart, (list, tuple)):
                patt = [Chem.MolFromSmarts(s) for s in smart]
            else:
                patt = Chem.MolFromSmarts(smart)
            obj.smart_rdkit = patt

        key = obj.group_id
        if isinstance(patt, (list, tuple)):
            hits = set()
            for p in patt:
                hits.update(list(rdkitmol.GetSubstructMatches(p)))
            hits = list(hits)
        else:
            hits = list(rdkitmol.GetSubstructMatches(patt))
            if not hits and len(obj.atoms) == 1 and 'H' in obj.atoms:
                hits = list(rdkitmol_Hs.GetSubstructMatches(patt))
                # Special handling for H2 molecule
                if hits:
                    # If this is a hydrogen-only group (like H2) and the molecule has only H atoms
                    num_atoms = rdkitmol_Hs.GetNumAtoms()
                    if num_atoms == H_count and H_count == obj.atoms['H']:
                        # For H2, return all expected values
                        counts = {key: 1}  # One instance of this group (H2)
                        group_assignments = {key: [()]}  # Empty tuple as there are no heavy atoms
                        matched_atoms = set()  # No heavy atoms to match
                        success = True
                        status = 'OK'
                        return counts, group_assignments, matched_atoms, success, status
        
        if hits:
            all_matches[key] = hits
            counts[key] = len(hits)

    # Higher should be lower
    priorities = [i.priority for i in catalog]
    groups = [i.group_id for i in catalog]
    group_to_obj = {o.group_id: o for o in catalog}
    catalog_by_priority =  [group_to_obj[g] for _, g in sorted(zip(priorities, groups), reverse=True)]

    all_heavies_matched_by_a_pattern = set()
    for v in all_matches.values():
        for t in v:
            all_heavies_matched_by_a_pattern.update(t)

    # excludes H
    DEBUG_VISUALIZATION = False
    if DEBUG_VISUALIZATION:
        from rdkit.Chem import Draw
        import matplotlib.pyplot as plt
        from matplotlib import colors
        
        # First, count total number of matches across all patterns
        total_matches = sum(len(matches) for matches in all_matches.values())
        
        if total_matches > 0:
            # Calculate grid dimensions - still use 3 columns
            n_cols = 3
            n_rows = (total_matches + n_cols - 1) // n_cols
            # Calculate figure size - adjust these multipliers as needed
            fig_width = n_cols * 2.2  # 5 inches per column
            fig_height = n_rows * 2.2  # 5 inches per row
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            if n_rows * n_cols > 1:
                axes = axes.flatten()
            else:
                axes = [axes]
                
            # Single highlight color since each match gets its own subplot
            highlight_color = (0.678, 0.847, 0.902)  # lightblue
            
            current_ax_idx = 0
            for group_id, matches in all_matches.items():
                pattern_obj = group_to_obj[group_id]
                
                # Make a subplot for each individual match
                for match_idx, match in enumerate(matches):
                    if current_ax_idx < len(axes):
                        ax = axes[current_ax_idx]
                        
                        # Create a copy of the molecule for this visualization
                        mol_copy = Chem.Mol(rdkitmol)
                        
                        # Add atom indices as labels
                        for atom in mol_copy.GetAtoms():
                            atom.SetProp('atomLabel', str(atom.GetIdx()))
                        
                        # Create the image with this single match highlighted
                        img = Draw.MolToImage(mol_copy, 
                                            highlightAtoms=list(match),
                                            highlightColor=highlight_color)
                        
                        ax.imshow(img)
                        ax.axis('off')
                        
                        # Add pattern info as title
                        ax.set_title(f"Group {group_id}\nMatch {match_idx + 1}\nSMARTS: {pattern_obj.smarts}", 
                                fontsize=10, pad=10)
                        
                        current_ax_idx += 1
            
            # Remove empty subplots
            for idx in range(current_ax_idx, len(axes)):
                fig.delaxes(axes[idx])
                
            plt.tight_layout()
            plt.show()



    ignore_matches = set()
    matched_atoms, final_group_counts, final_assignments = run_match(catalog_by_priority, all_matches, ignore_matches, all_atom_idxs, H_count)
    # Count the hydrogens

    heavy_atom_matched = atom_count == len(matched_atoms)
    hydrogens_found = 0
    for found_group in final_assignments.keys():
        for found_atoms in final_assignments[found_group]:
            if group_to_obj[found_group].hydrogen_from_smarts:
                hydrogens_found += sum(H_counts_by_idx[i] for i in found_atoms)
            else:
                hydrogens_found += group_to_obj[found_group].atoms.get('H', 0)

    #hydrogens_found = sum(group_to_obj[g].atoms.get('H', 0)*v for g, v in final_group_counts.items())
    hydrogens_matched = hydrogens_found == H_count

    if len(all_heavies_matched_by_a_pattern) != atom_count:
        status = 'Did not match all atoms present'
        success = False
        return final_group_counts, final_assignments, matched_atoms, success, status

    success = heavy_atom_matched and hydrogens_matched
    if not success:
        things_to_ignore = []
        for k in all_matches:
            for v in all_matches[k]:
                things_to_ignore.append((k, v))

        # if len(things_to_ignore) < 25:
            # remove_up_to = 4
        # elif len(things_to_ignore) < 25:
        #     remove_up_to = 3
        # elif len(things_to_ignore) < 18:
        #     remove_up_to = 2
        # else:
            # remove_up_to = 3
        remove_up_to = 4
        max_tries = 5000
        tries = 0

        done = False
        for remove in range(1, remove_up_to+1):
            if done:
                break
            for ignore_matches in combinations(things_to_ignore, remove):
                tries += 1
                if tries > max_tries:
                    break

                ignore_matches = set(ignore_matches)
                matched_atoms, final_group_counts, final_assignments = run_match(catalog_by_priority, all_matches, ignore_matches, all_atom_idxs, H_count)
                heavy_atom_matched = atom_count == len(matched_atoms)
                if not heavy_atom_matched:
                    continue

                hydrogens_found = 0
                for found_group in final_assignments.keys():
                    for found_atoms in final_assignments[found_group]:
                        if group_to_obj[found_group].hydrogen_from_smarts:
                            hydrogens_found += sum(H_counts_by_idx[i] for i in found_atoms)
                        else:
                            hydrogens_found += group_to_obj[found_group].atoms.get('H', 0)

                hydrogens_matched = hydrogens_found == H_count
                success = heavy_atom_matched and hydrogens_matched

                if success:
                    done = True
                    break

    if not success:
        status = 'Did not match all atoms present'

    return final_group_counts, final_assignments, matched_atoms, success, status

def run_match(catalog_by_priority, all_matches, ignore_matches, all_atom_idxs,
              H_count):
    matched_atoms = set()
    final_group_counts = {}
    final_assignments = {}
    for obj in catalog_by_priority:
        if obj.group_id in all_matches:
            for match in all_matches[obj.group_id]:
                match_set = set(match)
                if match_set.intersection(matched_atoms):
                    # At least one atom is already matched - keep looking
                    continue

                # If the group matches everything, check the group has the right number of hydrogens
                if match_set == all_atom_idxs and H_count and obj.atoms.get('H', 0) != H_count:
                    continue

                if (obj.group_id, match) in ignore_matches:
                    continue
                matched_atoms.update(match)
                try:
                    final_group_counts[obj.group_id]
                except:
                    final_group_counts[obj.group_id] = 0
                final_group_counts[obj.group_id] += 1

                try:
                    final_assignments[obj.group_id]
                except:
                    final_assignments[obj.group_id] = []
                final_assignments[obj.group_id].append(match)
    return matched_atoms, final_group_counts, final_assignments



def smarts_fragment(catalog, rdkitmol=None, smi=None, deduplicate=True):
    r'''Fragments a molecule into a set of unique groups and counts as
    specified by the `catalog`. The molecule can either be an rdkit
    molecule object, or a smiles string which will be parsed by rdkit.
    Returns a dictionary of groups and their counts according to the
    indexes of the catalog provided.

    Parameters
    ----------
    catalog : dict
        Dictionary indexed by keys pointing to smarts strings, [-]
    rdkitmol : mol, optional
        Molecule as rdkit object, [-]
    smi : str, optional
        Smiles string representing a chemical, [-]

    Returns
    -------
    counts : dict
        Dictionaty of integer counts of the found groups only, indexed by
        the same keys used by the catalog [-]
    success : bool
        Whether or not molecule was fully and uniquely fragmented, [-]
    status : str
        A string holding an explanation of why the molecule failed to be
        fragmented, if it fails; 'OK' if it suceeds.

    Notes
    -----
    Raises an exception if rdkit is not installed, or `smi` or `rdkitmol` is
    not defined.

    Examples
    --------
    Acetone:

    >>> smarts_fragment(catalog=J_BIGGS_JOBACK_SMARTS_id_dict, smi='CC(=O)C') # doctest:+SKIP
    ({1: 2, 24: 1}, True, 'OK')

    Sodium sulfate, (Na2O4S):

    >>> smarts_fragment(catalog=J_BIGGS_JOBACK_SMARTS_id_dict, smi='[O-]S(=O)(=O)[O-].[Na+].[Na+]') # doctest:+SKIP
    ({29: 4}, False, 'Did not match all atoms present')

    Propionic anhydride (C6H10O3):

    >>> smarts_fragment(catalog=J_BIGGS_JOBACK_SMARTS_id_dict, smi='CCC(=O)OC(=O)CC') # doctest:+SKIP
    ({1: 2, 2: 2, 28: 2}, False, 'Matched some atoms repeatedly: [4]')
    '''
    if not loaded_rdkit:
        load_rdkit_modules()
    if rdkitmol is None and smi is None:
        raise Exception('Either an rdkit mol or a smiles string is required')
    if smi is not None:
        rdkitmol = Chem.MolFromSmiles(smi)
        if rdkitmol is None:
            status = 'Failed to construct mol'
            success = False
            return {}, success, status
    from collections import Counter

    atom_count = len(rdkitmol.GetAtoms())
    status = 'OK'
    success = True

    counts = {}
    all_matches = {}
    for key, smart in catalog.items():
        if isinstance(smart, str):
            patt = Chem.MolFromSmarts(smart)
        else:
            patt = smart
        hits = list(rdkitmol.GetSubstructMatches(patt))
        if hits:
            all_matches[key] = hits
            counts[key] = len(hits)

    # Duplicate group cleanup
    matched_atoms = []
    for i in all_matches.values():
        for j in i:
            matched_atoms.extend(j)

    if deduplicate:
        dups = [i for i, c in Counter(matched_atoms).items() if c > 1]
        iteration = 0
        while (dups and iteration < 100):
            dup = dups[0]

            dup_smart_matches = []
            for group, group_match_list in all_matches.items():
                for i, group_match_i in enumerate(group_match_list):
                    if dup in group_match_i:
                        dup_smart_matches.append((group, i, group_match_i, len(group_match_i)))


            sizes = [i[3] for i in dup_smart_matches]
            max_size = max(sizes)
#            print(sizes, 'sizes', 'dup', dup, 'working_data', dup_smart_matches)
            if sizes.count(max_size) > 1:
                iteration += 1
#                print('BAD')
                # Two same size groups, continue, can't do anything
                continue
            else:
                # Remove matches that are not the largest
                for group, idx, positions, size in dup_smart_matches:
                    if size != max_size:
                        # Not handling the case of multiple duplicate matches right, indexes changing!!!
                        del all_matches[group][idx]
                        continue

            matched_atoms = []
            for i in all_matches.values():
                for j in i:
                    matched_atoms.extend(j)

            dups = [i for i, c in Counter(matched_atoms).items() if c > 1]
            iteration += 1

    matched_atoms = set()
    for i in all_matches.values():
        for j in i:
            matched_atoms.update(j)
    if len(matched_atoms) != atom_count:
        status = 'Did not match all atoms present'
        success = False

    # Check the atom aount again, this time looking for duplicate matches (only if have yet to fail)
    if success:
        matched_atoms = []
        for i in all_matches.values():
            for j in i:
                matched_atoms.extend(j)
        if len(matched_atoms) < atom_count:
            status = 'Matched %d of %d atoms only' %(len(matched_atoms), atom_count)
            success = False
        elif len(matched_atoms) > atom_count:
            status = 'Matched some atoms repeatedly: %s' %( [i for i, c in Counter(matched_atoms).items() if c > 1])
            success = False

    return counts, success, status

