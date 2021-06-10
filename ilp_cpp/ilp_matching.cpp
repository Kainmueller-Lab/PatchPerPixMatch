#define HAVE_GUROBI
// get code from: https://github.com/funkey/solvers
// config.h GurobiBackend.cpp GurobiBackend.h  LinearConstraint.cpp LinearConstraint.h LinearConstraints.cpp LinearConstraints.h LinearObjective.h LinearSolverBackendFactory.h LinearSolverBackend.h QuadraticObjective.cpp QuadraticObjective.h QuadraticSolverBackendFactory.h QuadraticSolverBackend.h QuadraticSolverParameters.h Relation.h Sense.h Solution.cpp Solution.h VariableType.h
#include "GurobiBackend.h"
#include "LinearObjective.h"
#include "LinearConstraint.h"
#include "LinearConstraints.h"
#include "Relation.h"
#include "Solution.h"
#include "VariableType.h"
#include <vector>
#include <tuple>
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <chrono>

#include <algorithm>
#include <numeric>
#include <span>
#include <ranges>
#include <cstdint>

std::vector<size_t> argsort(float* array, int sz) {
  std::vector<size_t> indices(sz);
   std::iota(indices.begin(), indices.end(), 0);
   std::sort(indices.begin(), indices.end(),
	     [&array](int left, int right) -> bool {
	       // sort indices according to corresponding array element
	       return array[left] < array[right];
	     });

   return indices;
 }

std::vector<size_t> argpartition(float* array, int sz, int nth) {
  std::vector<size_t> indices(sz);
  std::iota(indices.begin(), indices.end(), 0);

  std::ranges::nth_element(indices, indices.begin() + nth, [&array](int left, int right) -> bool {
    // sort indices according to corresponding array element
    return array[left] < array[right];
  });

  return indices;
}


GurobiBackend* solver;
extern "C"
void make_solver() {
  solver = new GurobiBackend;
}

extern "C"
void delete_solver() {
  delete solver;
}


extern "C"
int solve(float* dists, int num_em, int num_frag, float max_dist,
	  double* solution, int max_matches) {

  // auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::tuple<int, int, int>> edges;
  std::vector<std::vector<int>> fp_to_edges(num_frag, std::vector<int>{});
  std::vector<std::vector<int>> ep_to_edges(num_em, std::vector<int>{});

  int nth = 25;

  int num_vars = num_em + num_frag;
  for (int ei = 0; ei < num_em; ++ei) {
    int idx = ei * num_frag;
    if (num_frag < nth) {
      for (int fi = 0; fi < num_frag; ++fi) {
	if (dists[idx+fi] > max_dist) {
	  continue;
	}
	edges.emplace_back(num_vars, ei, fi);
	ep_to_edges[ei].push_back(num_vars);
	fp_to_edges[fi].push_back(num_vars);
	++num_vars;
      }
    }
    else {
      auto sorted_ids = argpartition(&dists[ei*num_frag], num_frag, nth);
      for (int fi = 0; fi < std::min(nth, num_frag); ++fi) {
	int fidx = sorted_ids[fi];
	if (dists[idx + fidx] > max_dist) {
	  continue;
	}
	edges.emplace_back(num_vars, ei, fidx);
	ep_to_edges[ei].push_back(num_vars);
	fp_to_edges[fidx].push_back(num_vars);
	++num_vars;
      }
    }
  }

  if (edges.size() == 0) {
    return 1;
  }

  int num_edges = edges.size();
  LinearObjective obj(num_vars);
  for (int edi = 0; edi < num_edges; ++edi) {
    int idx = std::get<1>(edges[edi]) * num_frag + std::get<2>(edges[edi]);
    obj.setCoefficient(std::get<0>(edges[edi]),
		       dists[idx]-max_dist);
  }


  solver->initialize(num_vars, Binary);
  solver->setObjective(obj);

  LinearConstraints consts;

  int fp_size = fp_to_edges.size();
  for (int fp = 0; fp < fp_size; ++fp) {
    LinearConstraint cnst2;
    cnst2.setCoefficient(fp+num_em, max_matches);
    for (int e = 0; e < fp_to_edges[fp].size(); ++e) {
      cnst2.setCoefficient(fp_to_edges[fp][e], -1);
    }
    cnst2.setRelation(GreaterEqual);
    cnst2.setValue(0);
    solver->addConstraint(cnst2);
  }

  int ep_size = ep_to_edges.size();
  for (int ep = 0; ep < ep_size; ++ep) {
    LinearConstraint cnst3;
    cnst3.setCoefficient(ep, -1);
    for (int e = 0; e < ep_to_edges[ep].size(); ++e) {
      cnst3.setCoefficient(ep_to_edges[ep][e], 1);
    }
    cnst3.setRelation(Equal);
    cnst3.setValue(0);
    solver->addConstraint(cnst3);
  }

  solver->setTimeout(120);
  solver->setNumThreads(1);

  Solution sol(num_vars);

  // auto mid = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
  // std::cout << "setting up " << duration.count() << std::endl;
  std::string message;
  // std::cout << "solving" << std::endl;
  int ret = solver->solve(sol, message);

  // auto end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
  // std::cout << "solving " << duration.count() << std::endl;
  memcpy(solution, (void*)sol.getVector().data(),
	 sizeof(double)*num_em);
  // std::cout << "done" << std::endl;

  return ret;
}


extern "C"
int solve2(double* dists, int num_em, int num_frag, float max_dist,
	   std::int64_t* inds, int* num_neigh,
	   double* solution,  int max_matches) {
  // auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::tuple<int, int, int, int>> edges;
  std::vector<std::vector<int>> fp_to_edges(num_frag, std::vector<int>{});
  std::vector<std::vector<int>> ep_to_edges(num_em, std::vector<int>{});


  int num_vars_start = num_em + num_frag;
  int num_vars = num_vars_start;
  int idx = 0;

  for (int fi = 0; fi < num_frag; ++fi) {
    for (int n = 0; n < num_neigh[fi]; ++n) {
      int ei = inds[idx];
      edges.emplace_back(num_vars, ei, fi, idx);
      ep_to_edges[ei].push_back(num_vars);
      fp_to_edges[fi].push_back(num_vars);
      ++num_vars;
      ++idx;
    }
  }

  if (edges.size() == 0) {
    return 1;
  }

  int num_edges = edges.size();
  LinearObjective obj(num_vars);
  for (int edi = 0; edi < num_edges; ++edi) {
    int idx = std::get<3>(edges[edi]);
    obj.setCoefficient(std::get<0>(edges[edi]),
		       dists[idx]-max_dist);
  }

  solver->initialize(num_vars, Binary);
  solver->setObjective(obj);

  LinearConstraints consts;

  int fp_size = fp_to_edges.size();
  for (int fp = 0; fp < fp_size; ++fp) {
    LinearConstraint cnst2;
    cnst2.setCoefficient(fp+num_em, max_matches);
    for (int e = 0; e < fp_to_edges[fp].size(); ++e) {
      cnst2.setCoefficient(fp_to_edges[fp][e], -1);
    }
    cnst2.setRelation(GreaterEqual);
    cnst2.setValue(0);
    solver->addConstraint(cnst2);
  }

  int ep_size = ep_to_edges.size();
  for (int ep = 0; ep < ep_size; ++ep) {
    LinearConstraint cnst3;
    cnst3.setCoefficient(ep, -1);
    for (int e = 0; e < ep_to_edges[ep].size(); ++e) {
      cnst3.setCoefficient(ep_to_edges[ep][e], 1);
    }
    cnst3.setRelation(Equal);
    cnst3.setValue(0);
    solver->addConstraint(cnst3);
  }

  solver->setTimeout(120);
  solver->setNumThreads(1);

  Solution sol(num_vars);

  // auto mid = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
  // std::cout << "setting up, took " << duration.count() << std::endl;
  std::string message;
  // std::cout << "solving" << std::endl;
  int ret = solver->solve(sol, message);
  // int ret = solver.solve(sol, message);

  // auto end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
  // std::cout << "solvingcpp, " << duration.count() << std::endl;
  memcpy(solution, (void*)sol.getVector().data(),
	 sizeof(double)*num_em);

  return ret;
}


// solve all fragments in one go
extern "C"
int solve3(double* dists, int num_em,
	   int num_frags, int num_fp_total, int* num_fp_per_frag,
	   float max_dist, std::int64_t* inds, int* num_neigh,
	   double* solution, int max_matches) {

  std::vector<std::tuple<int, int, int, int, int>> edges;
  std::vector<std::vector<int>> fp_to_edges(num_fp_total, std::vector<int>{});
  std::vector<std::vector<std::vector<int>>> ep_to_edges_per_frag(
								  num_frags,
	std::vector<std::vector<int>>(num_em,
				     std::vector<int>{}));

  // auto start = std::chrono::high_resolution_clock::now();

  int num_vars_start = num_em * num_frags + num_fp_total;
  int num_vars = num_vars_start;
  int didx = 0;
  int nidx = 0;

  for (int frag = 0; frag < num_frags; ++frag) {
    for (int fi = 0; fi < num_fp_per_frag[frag]; ++fi) {
      for (int n = 0; n < num_neigh[nidx]; ++n) {
	int ei = inds[didx];
	edges.emplace_back(num_vars, ei, nidx, didx, frag);
	ep_to_edges_per_frag[frag][ei].push_back(num_vars);
	fp_to_edges[nidx].push_back(num_vars);
	++num_vars;
	++didx;
      }
      ++nidx;
    }
  }

  if (edges.size() == 0) {
    return 1;
  }

  int num_edges = edges.size();
  LinearObjective obj(num_vars);
  for (int edi = 0; edi < num_edges; ++edi) {
    int didx = std::get<3>(edges[edi]);
    obj.setCoefficient(std::get<0>(edges[edi]),
		       dists[didx]-max_dist);
  }

  solver->initialize(num_vars, Binary);
  solver->setObjective(obj);

  LinearConstraints consts;

  int fp_size = fp_to_edges.size();
  for (int fp = 0; fp < fp_size; ++fp) {
    LinearConstraint cnst2;
    cnst2.setCoefficient(fp + num_em * num_frags, max_matches);
    for (int e = 0; e < fp_to_edges[fp].size(); ++e) {
      cnst2.setCoefficient(fp_to_edges[fp][e], -1);
    }
    cnst2.setRelation(GreaterEqual);
    cnst2.setValue(0);
    solver->addConstraint(cnst2);
  }

  for (int frag = 0; frag < num_frags; ++frag) {
    int ep_size = ep_to_edges_per_frag[frag].size();
    for (int ep = 0; ep < ep_size; ++ep) {
      LinearConstraint cnst3;
      cnst3.setCoefficient(num_em * frag + ep, -1);
      for (int e = 0; e < ep_to_edges_per_frag[frag][ep].size(); ++e) {
	cnst3.setCoefficient(ep_to_edges_per_frag[frag][ep][e], 1);
      }
      cnst3.setRelation(Equal);
      cnst3.setValue(0);
      solver->addConstraint(cnst3);
    }
  }

  solver->setTimeout(120);
  solver->setNumThreads(1);

  Solution sol(num_vars);

  // auto mid = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
  // std::cout << "setting up, took " << duration.count() << std::endl;
  std::string message;
  // std::cout << "solving" << std::endl;
  int ret = solver->solve(sol, message);

  // auto end = std::chrono::high_resolution_clock::now();
  // duration = std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
  // std::cout << "solvingcpp, took " << duration.count() << std::endl;
  memcpy(solution, (void*)sol.getVector().data(),
	 sizeof(double)*num_em*num_frags);

  return ret;
}
