print("Importing modules")
import pandas as pd
import numpy as np
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import matching_tools
from tqdm import tqdm
import os
import json
import event_tree

import sys

parquet_path = "/vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/NMSSM_Y_gg_Low_Mass/Oct22/DY_Diboson_Diphoton_StandardLowSelection/merged/merged_nominal.parquet"
summary_path = "/vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/NMSSM_Y_gg_Low_Mass/Oct22/DY_Diboson_Diphoton_StandardLowSelection/merged/summary.json"
root_file_dir = "CMSSW_10_6_25/src/PhysicsTools/NanoAODTools/output/ElectronVetoNov23"
event_classification_json_file = "event_classification_ElectronVeto_test.json"

# parquet_path = "/vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/NMSSM_Y_gg_Low_Mass/Feb23/24Feb2023_relic_DY_study_diEle_HLT/merged_nominal.parquet"
# summary_path = "/vols/cms/mdk16/ggtt/Inputs/HiggsDNA/pre/NMSSM_Y_gg_Low_Mass/Feb23/24Feb2023_relic_DY_study_diEle_HLT/summary.json"
# root_file_dir = "CMSSW_10_6_25/src/PhysicsTools/NanoAODTools/output/InvElectronVetoNov23"
# event_classification_json_file = "event_classification_InvElectronVeto.json"

print("Loading Parquet")
df = pd.read_parquet(parquet_path)
with open(summary_path, "r") as f:
  proc_dict = json.load(f)["sample_id_map"]
for proc in proc_dict.keys():
  if "DY" in proc:
    DY_id = proc_dict[proc]
df = df[df.process_id == DY_id]

print(len(df))

years = ["2016", "2016APV", "2017", "2018"]
if len(sys.argv) > 1:
  years = [years[int(sys.argv[1])]]
  print(f"Doing {years}")
  event_classification_json_file += ("." + years[0])
year_dict = {
  "2016": b"2016UL_pos",
  "2016APV": b"2016UL_pre",
  "2017": b"2017",
  "2018": b"2018"
}

category_key = {
  1: ["Tau", "Muon"],
  2: ["Tau", "Electron"],
  3: ["Tau", "Tau"],
  4: ["Muon", "Muon"],
  5: ["Electron", "Electron"],
  6: ["Muon", "Electron"],
  7: ["Tau", "IsoTrack"],
  8: ["Tau", ""]
}

id_to_object = {
  11: "Electron",
  13: "Muon",
  15: "Tau",
  1: "IsoTrack"
}

df_years = []

event_classification = {cat:[] for cat in range(1,9)}

event_classification_to_save = {year:{} for year in years}

for year in years:
  print("-"*40)
  print(f"Looking at {year}")
  df_year = df[df.year == year_dict[year]]

  print(len(df_year))
  
  print("Loading root file")
  fname = os.path.join(root_file_dir, year, "merged.root")
  events = NanoEventsFactory.from_root(
    fname,
    schemaclass=NanoAODSchema.v7,
    metadata={"dataset": "DYJets"},
  ).events()

  print(len(df_year), len(events.event), len(set(df_year.event).union(events.event)))
  events = events[np.isin(np.array(events.event), np.array(df_year.event))]
  df_year = df_year[np.isin(np.array(df_year.event), np.array(events.event))]
  print(len(df_year), len(events.event), len(set(df_year.event).union(events.event)))

  assert len(df_year) == len(events.event)
  if (np.array(np.sort(df_year.event)) == np.array(np.sort(events.event))).all():
    print("IDs match")
  else:
    raise Exception("IDs do not match")

  df_year.sort_values("event", inplace=True)
  events = events[np.argsort(events.event)]

  print(df_year.event)
  print(events.event)

  truth_ids = []
  
  for i in tqdm(range(len(df_year)), disable=False):
    row = df_year.iloc[i]
    # if row.category == 8:
    #   continue
    #print("\nEvent %d, category %d"%(i, row.category))
    #print(row.Diphoton_mass)

    #if (row.LeadPhoton_pixelSeed) | (row.SubleadPhoton_pixelSeed):
    #  continue

    #print(df_year.LeadPhoton_pt.quantile(q=0.75))
    # if row.LeadPhoton_pt < df_year.LeadPhoton_pt.median():
    #   continue

    # if abs(row.lead_lepton_id) == 15:
    #   tau = matching_tools.findNanoAODObject(events[i], "Tau", row.lead_lepton_eta, row.lead_lepton_phi)
    #   #print(tau.idDeepTau2017v2p1VSjet)
    #   if tau.idDeepTau2017v2p1VSjet & 16 == 0:
    #     #print("skipping")
    #     continue
    # if abs(row.sublead_lepton_id) == 15:
    #   tau = matching_tools.findNanoAODObject(events[i], "Tau", row.sublead_lepton_eta, row.sublead_lepton_phi)
    #   #print(tau.idDeepTau2017v2p1VSjet)
    #   if tau.idDeepTau2017v2p1VSjet & 16 == 0:
    #     #print("skipping")
    #     continue

    classification = matching_tools.classifyEvent(events[i], row.LeadPhoton_eta, row.LeadPhoton_phi, row.SubleadPhoton_eta, row.SubleadPhoton_phi)
    #classification = 2
    # if (row.category==3) and (classification==1):
    #   print("\nEvent %d, category %d"%(i, row.category))
    #print(classification)
    event_classification[row.category].append(classification)
    event_classification_to_save[year][str(row.event)] = classification

    #continue

    lp_gen_idx, lp_pdg = matching_tools.getMatchingInfo(events[i], "Photon", row.LeadPhoton_eta, row.LeadPhoton_phi)
    slp_gen_idx, slp_pdg = matching_tools.getMatchingInfo(events[i], "Photon", row.SubleadPhoton_eta, row.SubleadPhoton_phi)
    ll_gen_idx, ll_pdg = matching_tools.getMatchingInfo(events[i], id_to_object[abs(row.lead_lepton_id)], row.lead_lepton_eta, row.lead_lepton_phi)   
    if row.category != 8:
      sll_gen_idx, sll_pdg = matching_tools.getMatchingInfo(events[i], id_to_object[abs(row.sublead_lepton_id)], row.sublead_lepton_eta, row.sublead_lepton_phi)
    else:
      sll_gen_idx, sll_pdg = -999, -999
    truth_ids.append([lp_gen_idx, slp_gen_idx, ll_gen_idx, sll_gen_idx])
    groups = [[lp_gen_idx, slp_gen_idx], [], [], [], []] #groups for hadronic tau, muon, electron
    
    ordering = {"Tau":1, "Muon":2, "Electron":3, "IsoTrack":4}
    groups[ordering[id_to_object[abs(row.lead_lepton_id)]]].append(ll_gen_idx)
    if row.category != 8:
      groups[ordering[id_to_object[abs(row.sublead_lepton_id)]]].append(sll_gen_idx)

    savedir = "event_graphs/class%s/%s"%(classification, "_".join(category_key[row.category]))
    os.makedirs(savedir, exist_ok=True)
    event_tree.saveGraph(events[i], "%s/year_%s_event_%d.png"%(savedir, year, i), groups=groups, prune=True, mgg=row.Diphoton_mass)
    event_tree.saveGraph(events[i], "%s/year_%s_event_%d_unpruned.png"%(savedir, year, i), groups=groups, prune=False, mgg=row.Diphoton_mass)

  for cat, res in event_classification.items():
    print(cat)
    print(np.unique(res, return_counts=True))
  
  # truth_ids = np.array(truth_ids)

  # df_year["LeadPhoton_GenPart_id"] = truth_ids[:,0]
  # df_year["SubleadPhoton_GenPart_id"] = truth_ids[:,1]
  # df_year["lead_lepton_GenPart_id"] = truth_ids[:,2]
  # df_year["sublead_lepton_GenPart_id"] = truth_ids[:,3]

  # df_years.append(df_year)

import json
with open(event_classification_json_file, "w") as f:
  json.dump(event_classification_to_save, f)

# df_matched = pd.concat(df_years)
# df_matched.to_parquet("df_matched_test.parquet")
