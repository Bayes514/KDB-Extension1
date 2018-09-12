/* Open source system for classification learning from very large data
 ** Copyright (C) 2012 Geoffrey I Webb
 **
 ** This program is free software: you can redistribute it and/or modify
 ** it under the terms of the GNU General Public License as published by
 ** the Free Software Foundation, either version 3 of the License, or
 ** (at your option) any later version.
 **
 ** This program is distributed in the hope that it will be useful,
 ** but WITHOUT ANY WARRANTY; without even the implied warranty of
 ** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 ** GNU General Public License for more details.
 **
 ** You should have received a copy of the GNU General Public License
 ** along with this program. If not, see <http://www.gnu.org/licenses/>.
 **
 ** Please report any bugs to Geoff Webb <geoff.webb@monash.edu>
 */
#pragma once

#include "incrementalLearner.h"
#include "xxyDist.h"
#include <limits>
#include "crosstab.h"
#include "utils.h"

class TAN_IM6: public IncrementalLearner {
public:
	TAN_IM6();
	TAN_IM6(char* const *& argv, char* const * end);
	~TAN_IM6(void);

	void reset(InstanceStream &is);   ///< reset the learner prior to training
	void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
	void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
	void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
	void getCapabilities(capabilities &c);
	void classify_local(const instance &inst, std::vector<unsigned int> &parrent, std::vector<double> &classDist);
	double Cross_Entropy(std::vector<double> cd_1,std::vector<double> cd_2);
	void remodel(std::vector<unsigned int> &parrent,crosstab<float> &cmi);
	virtual void classify(const instance &inst, std::vector<double> &classDist);

private:
	unsigned int noCatAtts_;          ///< the number of categorical attributes.
	unsigned int noClasses_;                          ///< the number of classes

	InstanceStream* instanceStream_;
	std::vector<std::vector<CategoricalAttribute> > parents_y;
	std::vector<std::vector<std::vector<double> > > Iy;
	xxyDist xxyDist_;

	bool trainingIsFinished_; ///< true iff the learner is trained
	int threshold;
	const static CategoricalAttribute NOPARENT = 0xFFFFFFFFUL; //使用printf("%d",0xFFFFFFFFUL);输出是-1 cannot use std::numeric_limits<categoricalAttribute>::max() because some compilers will not allow it here
};
