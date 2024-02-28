"""
Given a object from a dataframe I need to do:
1) Find the reconstructed object in the nanoAOD
2) If there is a gen-match:
  i) Follow that chain all the way back
  ii) Also return the most relevant parent (first "isHardProcess")
3) If there is not a gen-match:
  i) Search all gen parts from a potential match
"""

import numpy as np

dphi = lambda x, y: abs(x-y) - 2*(abs(x-y) - np.pi) * (abs(x-y) // np.pi)

def getFlags(statusFlags):
    flag_dict = {0b1 : "isPrompt", 0b10 : "isDecayedLeptonHadron", 0b100 : "isTauDecayProduct", 
                0b1000 : "isPromptTauDecayProduct", 0b10000 : "isDirectTauDecayProduct", 
                0b100000 : "isDirectPromptTauDecayProduct", 0b1000000 : "isDirectHadronDecayProduct", 
                0b10000000 : "isHardProcess", 0b100000000 : "fromHardProcess", 
                0b1000000000 : "isHardProcessTauDecayProduct", 0b10000000000 : "isDirectHardProcessTauDecayProduct", 
                0b100000000000 : "fromHardProcessBeforeFSR", 0b1000000000000 : "isFirstCopy", 
                0b10000000000000 : "isLastCopy", 0b100000000000000 : "isLastCopyBeforeFSR"}
    flags = []
    for each in flag_dict.items():
      if (statusFlags & each[0])>0:
        flags.append(each[1])
    return flags

def findNanoAODObject(event, collection, eta, phi):
  for obj in event[collection]:
    dR = np.sqrt( (obj.eta - eta)**2 + dphi(obj.phi, phi)**2 )
    if dR < 0.001:
      return obj

def getMotherChain(event, idx):
  """Obj is a (or list of) GenPart idx"""

  if type(idx) != list:
    idx = [idx]

  mother_idx = event.GenPart[idx[-1]].genPartIdxMother
  if mother_idx == -1:
    return idx[::-1]
  else:
    idx.append(mother_idx)
    return getMotherChain(event, idx)
  
def getLastHardObject(event, chain):
  """Return idx associated with last object in chain which is part of the hard process"""  
  for idx in chain[::-1]:
    flags = getFlags(event.GenPart[idx].statusFlags)
    mother_idx = event.GenPart[idx].genPartIdxMother
    if ("isHardProcess" in flags) or (mother_idx == -1):
      #if mother_idx == -1: print("No hard object")
      return idx

def searchGenPartMatch(event, eta, phi):
  dR = []
  
  for obj in event.GenPart:
    if obj.pt == 0:
      dR.append(999)
    else:
      dR.append(np.sqrt( (obj.eta - eta)**2 + dphi(obj.phi, phi)**2 ))
  
  idx_min = np.argmin(dR)
  return idx_min, dR[idx_min]

def getGenPartIdx(event, nano_obj):
  genPartIdx = nano_obj.genPartIdx

  # check that the match makes sense
  if genPartIdx != -1:
    gen_part = event.GenPart[genPartIdx]
    dR = np.sqrt( (nano_obj.eta - gen_part.eta)**2 + dphi(nano_obj.phi, gen_part.phi)**2 )
    
    #if doesn't make sense, return no match
    if dR > 0.4:
      print("Match does not make sense")
      genPartIdx = -1

  if genPartIdx == -1:
    genPartIdx, dR = searchGenPartMatch(event, nano_obj.eta, nano_obj.phi)
    if dR > 0.4:
      genPartIdx = -1

  return genPartIdx

def findClosestTau(event, chain):
  for idx in chain[::-1]:
    if abs(event.GenPart[idx].pdgId) == 15:
      return idx
  return chain[-1] # if not find Tau, just use matched GenPart

def getMatchingInfo(event, collection, eta, phi):
  """Find gen match and return genPartIdx and pdgid"""
  nano_obj = findNanoAODObject(event, collection, eta, phi)

  genPartIdx = -1
  if collection != "IsoTrack":
    genPartIdx = getGenPartIdx(event, nano_obj)

  if genPartIdx == -1:
    genPartIdx, dR = searchGenPartMatch(event, nano_obj.eta, nano_obj.phi)

  # if still no match
  if genPartIdx == -1:
    return -1, -999
  else:
    chain = getMotherChain(event, genPartIdx)
    last_hard_object_idx = getLastHardObject(event, chain) #

    # sometimes matched particle is not the originating tau, so we search up the chain
    if collection == "Tau":
      genPartIdx = findClosestTau(event, chain)

    return genPartIdx, event.GenPart[last_hard_object_idx].pdgId

# def classifyEvent(event, eta1, phi1, eta2, phi2):
#   """
#   Decide if an event is peaking or not
#   Output:
#   0 = Not Peaking
#   1 = Peaking
#   2 = Inconclusive (for e.g. not good gen matching)

#   eta(1/2) and phi(1/2) are the eta and phi of the two reconstructed photons

#   An event is deemed as peaking if
#   1) The reconstructed photons are gen matched to children of a Z and if
#   2) If there is a gen matched photon, it overlaps with an electron from the Z (overlap means dR < 0.4)
#   """

#   # find reco-level objects
#   eta, phi = [eta1, eta2], [phi1, phi2]
#   photons = [findNanoAODObject(event, "Photon", eta[i], phi[i]) for i in range(2)]

#   # find gen-level matches
#   photon_gen_idx = []
#   for photon in photons:
#     genPartIdx = getGenPartIdx(event, photon)
#     photon_gen_idx.append(genPartIdx)

#   # return inconclusive result if one of the photons is not matched
#   if -1 in photon_gen_idx:
#     return 2

#   photons = [event.GenPart[idx] for idx in photon_gen_idx]

#   # check Z is in chain of both
#   chains = [getMotherChain(event, idx) for idx in photon_gen_idx]
#   pdg_chains = [[event.GenPart[idx].pdgId for idx in chain] for chain in chains]
#   if (23 not in pdg_chains[0]) or (23 not in pdg_chains[1]):
#     return 0
#   # check if tau in chain
#   if (15 in pdg_chains[0]) or (-15 in pdg_chains[0]) or (15 in pdg_chains[1]) or (-15 in pdg_chains[1]):
#     return 0

#   # check this was a Z->ee decay
#   hard_objects = [event.GenPart[getLastHardObject(event, chain)] for chain in chains]
#   if (abs(hard_objects[0].pdgId) != 11) or (abs(hard_objects[1].pdgId) != 11):
#     return 0

#   # check for photon overlap
#   """
#   If photon is matched to gen-level photon, require that it is within 0.4 of an electron
#   from Z which is not matched to the other photon.
#   """
#   other_electrons = [electron for electron in hard_objects if electron. not in photons]
#   photon_matched_photons = [photon for photon in photons if photon.pdgId == 22]

#   for photon in photon_matched_photons: 
#     close_to_e = False
#     for e in other_electrons:
#       if np.sqrt( (photon.eta - e.eta)**2 + dphi(photon.phi, e.phi)**2 ) < 0.4:
#         print("is close to e")
#         close_to_e = True
    
#     if not close_to_e:
#       return 0    

#   # electrons = hard_objects
#   # for idx in photon_gen_idx:
#   #   part = event.GenPart[idx]
#   #   if part.pdgId == 22:
#   #     close_to_e = False
#   #     for e in electrons:
#   #       print(part.eta, e.eta, part.phi, e.phi)
#   #       if np.sqrt( (part.eta - e.eta)**2 + dphi(part.phi, e.phi)**2 ) < 0.4:
#   #         print("is close to e")
#   #         close_to_e = True

#   #     if not close_to_e:
#   #       return 0

#   # require mass of Z to be within 10 GeV
#   # Z = event.GenPart[electrons[0].genPartIdxMother]
#   # assert Z.pdgId == 23
#   # #print(Z.mass)
#   # if abs(Z.mass-91.18) > 10:
#   #   return 0

#   # if satisfy all conditions
#   #assert abs((photons[0]+photons[1]).mass - 90) < 10, (photons[0]+photons[1]).mass 
#   return 1

def classifyEvent(event, eta1, phi1, eta2, phi2):
  """
  Decide if an event is peaking or not
  Output:
  0 = Not Peaking
  1 = Peaking
  2 = Inconclusive (for e.g. not good gen matching)

  eta(1/2) and phi(1/2) are the eta and phi of the two reconstructed photons

  An event is deemed as peaking if
  1) The event contains a Z->ee decay 
  2) Both reco-photons overlap (dR<0.4) with one the electrons
  3) Both electrons overlap (dR<0.4) with one of the reco-photons
  """

  # find hard electrons
  electrons_idx = []
  for idx, part in enumerate(event.GenPart):
    # if hard electron
    if (abs(part.pdgId) == 11) and ((event.GenPart[idx].statusFlags & 0b10000000) > 0):
      electrons_idx.append(idx)

  # if both reco-photons can be matched to different hard electrons -> peak
  if len(electrons_idx) >=2:
    electrons = [event.GenPart[idx] for idx in electrons_idx]
    # photon 1
    dRs1 = [np.sqrt((e.eta - eta1)**2 + dphi(e.phi, phi1)**2 ) for e in electrons]
    # photon 2
    dRs2 = [np.sqrt((e.eta - eta2)**2 + dphi(e.phi, phi2)**2 ) for e in electrons]
    
    both_matched = (min(dRs1) < 0.1) and (min(dRs2) < 0.1)
    different_matches = (np.argmin(dRs1) != np.argmin(dRs2))

    if both_matched and different_matches:
      return 1      

  # if closest matched object is a child of a z -> classify as Z->eeg

  # find reco-level objects
  eta, phi = [eta1, eta2], [phi1, phi2]
  photons = [findNanoAODObject(event, "Photon", eta[i], phi[i]) for i in range(2)]
  photons_idx = [getGenPartIdx(event, photon) for photon in photons]

  if len(electrons_idx) >=2:
    # if matched to one hard electron and the gen match for reco-photon is a photon
    # -> Zeeg
    if (min(dRs1) < 0.1) and (photons_idx[1] != -1) and (event.GenPart[photons_idx[1]].pdgId == 22):
      return 3
    elif (min(dRs2) < 0.1) and (photons_idx[0] != -1) and (event.GenPart[photons_idx[0]].pdgId == 22):
      return 3

  # find gen-level matches and try to find Z in chain
  Zeeg_matched = []
  for idx in photons_idx:
    if idx == -1:
      return 0

    chain = getMotherChain(event, idx)
    pdg_chain = [event.GenPart[idx].pdgId for idx in chain]
    Zeeg_matched.append( (23 in pdg_chain) )
  
  if sum(Zeeg_matched) == 2:
    return 3
  else:
    return 0

  #return 1

# # check that reco-photons can be matched to the electrons
#   electrons = [event.GenPart[idx] for idx in electrons_idx]
#   # photon 1
#   dRs1 = [np.sqrt((e.eta - eta1)**2 + dphi(e.phi, phi1)**2 ) for e in electrons]
#   # photon 2
#   dRs2 = [np.sqrt((e.eta - eta2)**2 + dphi(e.phi, phi2)**2 ) for e in electrons]
  
#   if (min(dRs1) < 0.4) and (min(dRs2) < 0.4):
#     return 1

#   # one final check: if closest matched object is not child of Z -> is non-peaking

#   # find reco-level objects
#   eta, phi = [eta1, eta2], [phi1, phi2]
#   photons = [findNanoAODObject(event, "Photon", eta[i], phi[i]) for i in range(2)]

#   # find gen-level matches and try to find Z in chain
#   for photon in photons:
#     idx = getGenPartIdx(event, photon)
#     if idx == -1:
#       return 0

#     chain = getMotherChain(event, idx)
#     pdg_chain = [event.GenPart[idx].pdgId for idx in chain]
#     if (23 not in pdg_chain) and ((15 not in pdg_chain) or (-15 in pdg_chain)):
#       return 0