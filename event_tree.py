import pydot
import particle
from copy import deepcopy
import numpy as np

#nodes take the first shape as default
shapes = ["circle", "box", "pentagon", "hexagon", "septagon", "octagon"] 

def getParticleName(pdgId):
  try:
    name = particle.Particle.from_pdgid(pdgId).name
  except particle.particle.particle.ParticleNotFound:
    name = "None"

  # if "(" in name:
  #   idx1 = name.index("(")
  #   idx2 = name.index(")")
  #   name = name[:idx1] + name[idx2+1:]
  return name

def createBasicGraph(event):
  nodes = [[-1, []] for each in event.GenPart]
  for idx, part in enumerate(event.GenPart):
    mother_idx = part.genPartIdxMother
    nodes[idx][0] = mother_idx
    nodes[mother_idx][1].append(idx)

  return nodes

def changeIdxInGroups(old_idx, new_idx, groups):
  for group in groups:
    for j, idx in enumerate(group):
      if idx == old_idx:
        group[j] = deepcopy(new_idx)

def pruneBasicGraph(event, basic_graph, groups=[], mgg=None):
  always_keep = [idx for group in groups for idx in group]

  # prune away soft photons
  for idx, node in enumerate(basic_graph):
    if (idx in always_keep) or (node[0] == None):
      continue

    elif (len(node[1]) == 0) and (event.GenPart[idx].pdgId == 22) and (event.GenPart[idx].pt < 1):
      basic_graph[node[0]][1].remove(idx) # remove this node from mother's children
      node[0] = None # remove link to mother


  for idx, node in enumerate(basic_graph):
    if node[0] == None:
      continue

    # remove gluons with no children
    elif (event.GenPart[idx].pdgId == 21) and (len(node[1]) == 0) and (idx not in always_keep):        
      basic_graph[node[0]][1].remove(idx) # remove this node from mother's children
      node[0] = None # remove link to mother

    # remove node where it has only one child of the same pdgId
    elif (len(node[1]) == 1) and (event.GenPart[idx].pdgId == event.GenPart[node[1][0]].pdgId):
      if idx in always_keep:
        changeIdxInGroups(idx, node[1][0], groups)
        always_keep[always_keep.index(idx)] = node[1][0]

      basic_graph[node[0]][1].remove(idx) # remove yourself from mother
      basic_graph[node[0]][1].append(deepcopy(node[1][0])) # give child to mother
      basic_graph[node[1][0]][0] = deepcopy(node[0]) # tell child its new mother

      # nullify node      
      node[0] = None
      node[1] = []


  # collect radiation and attach to mother
  for idx, node in enumerate(basic_graph):
    if (node[0] == None) or (idx in always_keep):
      continue
    
    pdgId = event.GenPart[idx].pdgId
    if (np.isin(abs(pdgId), [11, 13, 15])) and (len(node[1]) > 1):

      should_combine = True # only if children are same lepton type or photons and children don't have their own children
      for child_idx in node[1]:
        child_pdgId = event.GenPart[child_idx].pdgId
        right_types = np.isin(abs(child_pdgId), [abs(pdgId), 22])
        #have_children = len(basic_graph[child_idx][1]) > 0
        have_children = False
        
        if (not right_types) or have_children:
          should_combine = False
          break

      if should_combine:
        basic_graph[node[0]][1].remove(idx) # remove this node from its mother's children
        for child_idx in node[1]:
          basic_graph[child_idx][0] = deepcopy(node[0]) # tell children its new mother
          basic_graph[node[0]][1].append(deepcopy(child_idx)) # given children to mother

        # nullify node      
        node[0] = None
        node[1] = []

  return basic_graph

def createPydotGraph(event, basic_graph, groups=[], mgg=None):
  graph = pydot.Dot('my_graph', graph_type='graph', bgcolor='white')

  def getShape(idx):
    shape = shapes[0]
    for j, group in enumerate(groups):
      if idx in group:
        shape = shapes[j+1]
        break
    return shape
  
  graph.add_node(pydot.Node(-1, label="Init", shape=getShape(-1)))

  for idx, node in enumerate(basic_graph):
    if node[0] == None: # if pruned node
      continue

    part = event.GenPart[idx]
    name = getParticleName(part.pdgId)
    eta, phi = part.eta, part.phi
    if part.pt == 0:
      eta, phi = 0, 0
    #label = "%s\n%d\n%.2f\n%.0f\n%.2f, %.2f"%(name, part.status, part.pt, part.mass, eta, phi)
    #label = "%s\npt=%.0f\nm=%.0f"%(name, part.pt, part.mass)
    #label = "%s\n%d\npt=%.0f\nm=%.0f"%(name, idx, part.pt, part.mass)
    #label = "%s\n%d\n%.2f,%.2f\npt=%.0f\nm=%.0f"%(name, idx, part.eta, part.phi, part.pt, part.mass)
    label = "%s\n%.2f,%.2f\npt=%.0f\nm=%.0f"%(name, part.eta, part.phi, part.pt, part.mass)
    if name == "Z0":
      color = "crimson"
    else:
      color = "black"
    graph.add_node(pydot.Node(idx, label=label, shape=getShape(idx), color=color))

    graph.add_edge(pydot.Edge(node[0], idx))

  if mgg is not None:
    graph.add_node(pydot.Node(idx+1, label="mgg = %.2f"%mgg))

  return graph

def saveGraph(event, output_name, prune, **kwargs):
  basic_graph = createBasicGraph(event)
  if prune:
    basic_graph = pruneBasicGraph(event, basic_graph, **kwargs)
  graph = createPydotGraph(event, basic_graph, **kwargs)

  graph.write_png(output_name)
