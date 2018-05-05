import networkx as nx

def jaccard_wt(graph, node):
	nbrs = set(graph.neighbors(node))
	scores = []
    
	for n in graph.nodes():
		if n != node and not graph.has_edge(node,n):
			nbrs_b = set(graph.neighbors(n))
			nbrs_c = nbrs & nbrs_b
			n_sum = 0.0
			for val in nbrs_c:
				n_sum += 1.0/float(graph.degree(val))
				
			d_1 = 0.0
			d_2 = 0.0
		      
			for val in nbrs:
				d_1 += graph.degree(val)
            
			d_1 = 1/d_1  
			        
			for val in nbrs_b:
				d_2 += graph.degree(val)
				
			d_2 = 1/d_2
            
			score_temp = n_sum/(d_1+d_2)
            
			scores.append(((node,n),score_temp))
    
	return scores