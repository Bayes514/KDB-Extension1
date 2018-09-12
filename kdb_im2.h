/* Open source system for classification learning from very large data
 * Copyright (C) 2012 Geoffrey I Webb
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */

#pragma once

#include <limits.h>

#include "incrementalLearner.h"
#include "distributionTree.h"
#include "xxyDist.h"
#include "xxxyDist.h"
#include "yDist.h"



/**
<!-- globalinfo-start -->
 * Class for a k-dependence Bayesian classifier.<br/>
 * <br/>
 * For more information on k-dependence Bayesian classifiers, see:<br/>
 * <br/>
 * Sahami, M.: Learning limited dependence Bayesian classifiers. In: KDD-96:
 * Proceedings of the Second International Conference on Knowledge Discovery and
 * Data Mining, 335--338, 1996.
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * \@inproceedings{sahami1996learning,
 *   title={Learning limited dependence Bayesian classifiers},
 *   author={Sahami, M.},
 *   booktitle={KDD-96: Proceedings of the Second International Conference on
 *              Knowledge Discovery and Data Mining},
 *   pages={335--338},
 *   year={1996}
 * }
 * </pre>
 <!-- technical-bibtex-end -->
 *
 *
 * @author Geoff Webb (geoff.webb@monash.edu)
 */


class KDB_IM2 :  public IncrementalLearner
{
public:
  KDB_IM2();
  KDB_IM2(char*const*& argv, char*const* end);
  ~KDB_IM2(void);

  void reset(InstanceStream &is);   ///< reset the learner prior to training
  void initialisePass();            ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
  void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
  void finalisePass();              ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
  bool trainingIsFinished();        ///< true iff no more passes are required. updated by finalisePass()
  void getCapabilities(capabilities &c);
  double Cross_Entropy(std::vector<double> cd_1ocal,std::vector<double> cd_gen);
  virtual void classify(const instance &inst, std::vector<double> &classDist);
  void classify_local(const instance& inst, std::vector<distributionTree>  &dTree,std::vector<double> &posteriorDist);
  void classify_local(const instance& inst,std::vector<std::vector<CategoricalAttribute> > p,std::vector<double> &posteriorDist);


protected:
  unsigned int pass_;                                        ///< the number of passes for the learner
  unsigned int k_;                                           ///< the maximum number of parents
  unsigned int noCatAtts_;                                   ///< the number of categorical attributes.
  unsigned int noClasses_;                                   ///< the number of classes
  xxyDist dist_;                                             // used in the first pass
  xxxyDist xxxyDist_;
  yDist classDist_;                                          // used in the second pass and for classification
  std::vector< std::vector<distributionTree> > dTree_gens;                      // used in the second pass and for classification
  std::vector< std::vector<std::vector<CategoricalAttribute> > > parents_gens;

  InstanceStream* instanceStream_;
};
