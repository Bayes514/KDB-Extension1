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
#include "tan_im6.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>

 TAN_IM6::TAN_IM6() :
trainingIsFinished_(false)
{
}

TAN_IM6::TAN_IM6(char* const *&, char* const *) :
xxyDist_(), trainingIsFinished_(false)
{
    name_ = "TAN_IM6";
}

TAN_IM6::~TAN_IM6(void)
{
}

void TAN_IM6::reset(InstanceStream &is)
{
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    trainingIsFinished_ = false;

    //safeAlloc(parents, noCatAtts_);
    parents_y.resize(this->noClasses_);
    for(int i = 0;i<this->parents_y.size();i++){
        this->parents_y[i].resize(noCatAtts);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
        {
            this->parents_y[i][a] = NOPARENT;
        }
    }
    Iy.resize(this->noClasses_);
    xxyDist_.reset(is);
}

void TAN_IM6::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void TAN_IM6::initialisePass()
{
    assert(trainingIsFinished_ == false);
    //learner::initialisePass (pass_);
    //	dist->clear();
    //	for (CategoricalAttribute a = 0; a < meta->noCatAtts; a++) {
    //		parents_[a] = NOPARENT;
    //	}
}

void TAN_IM6::train(const instance &inst)
{
    xxyDist_.update(inst);
}



void TAN_IM6::classify_local(const instance &inst, std::vector<unsigned int> &parrent, std::vector<double> &classDist){
    for (CatValue y = 0; y < noClasses_; y++)
    {
        classDist[y] = xxyDist_.xyCounts.p(y);
    }
    int cur_best_y = 0;
    double cur_best_value = 0.0;
    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++)
    {
        const CategoricalAttribute parent = parrent[x1];

        if (parent == NOPARENT)
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist[y] *= xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
            }
        }
        else
        {
            for (CatValue y = 0; y < noClasses_; y++)
            {
                classDist[y] *= xxyDist_.p(x1, inst.getCatVal(x1), parent,
                        inst.getCatVal(parent), y);
            }
        }
    }
    /*
    for(int y =0;y<this->noClasses_;y++){
        if(classDist_cly[y] > cur_best_value){
            cur_best_value = classDist_cly[y];
            cur_best_y =y;
        }
    }
    */

    normalise(classDist);
}

double TAN_IM6:: Cross_Entropy(std::vector<double> cd_1ocal,std::vector<double> cd_gen){
    double result = 0.0;
    for(int y = 0;y<this->noClasses_;y++){
        result+=(
            - cd_gen[y] * log2(cd_1ocal[y])
        );
    }
    return result;
}
void TAN_IM6::classify(const instance &inst, std::vector<double> &classDist)
{
    // 第 local_y 个  模型
    std::vector<std::vector<unsigned int> > parrents_local_y;
    std::vector<double> classDist_locals;
    classDist_locals.resize(this->noClasses_);
    for(int y = 0;y<this->noClasses_;y++){
        classDist_locals[y] = 0.0;
    }
    parrents_local_y.resize(this->noClasses_);
    int model_selected = 0;
    double min_cross_entropy = 100000000000;
    for(int local_y = 0;local_y < this->noClasses_;local_y++){
        std::vector<std::vector<double > >Iy_local;
        Iy_local.resize(this->noCatAtts_);
        for(int i = 0;i<this->noCatAtts_;i++)
            Iy_local[i].resize(this->noCatAtts_);
        for(int xi = 0;xi<this->noCatAtts_;xi++){
            for(int xj =0;xj<this->noCatAtts_;xj++){
                if(xi == xj){
                    Iy_local[xi][xj] = 10000000;
                    continue;
                }
                else{
                    Iy_local[xi][xj] = 0.0;
                    Iy_local[xi][xj] +=(
                        (this->xxyDist_.jointP(xi,inst.getCatVal(xi),xj,inst.getCatVal(xj),local_y) / this->xxyDist_.xyCounts.p(local_y))
                                                                           *
                             log2(
                                     (this->xxyDist_.jointP(xi,inst.getCatVal(xi),xj,inst.getCatVal(xj),local_y) / this->xxyDist_.xyCounts.p(local_y))
                                                                           /
                                     (this->xxyDist_.xyCounts.p(xi,inst.getCatVal(xi),local_y) * this->xxyDist_.xyCounts.p(xj,inst.getCatVal(xj),local_y))
                                )
                        );
                }

            }
        }
        //建模 local

        parrents_local_y[local_y].resize(this->noCatAtts_);
        for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
        {
            parrents_local_y[local_y][a] = NOPARENT;
        }
        CategoricalAttribute firstAtt = 0;
        parrents_local_y[local_y][firstAtt] = NOPARENT;
        float *maxWeight;
        CategoricalAttribute *bestSoFar;
        CategoricalAttribute topCandidate = firstAtt;
        std::set<CategoricalAttribute> available;
        safeAlloc(maxWeight, noCatAtts_);
        safeAlloc(bestSoFar, noCatAtts_);
        maxWeight[firstAtt] = -std::numeric_limits<float>::max();

        for (CategoricalAttribute a = firstAtt + 1; a < noCatAtts_; a++)
        {
            maxWeight[a] = Iy_local[firstAtt][a];
            if (Iy_local[firstAtt][a] > maxWeight[topCandidate])
                topCandidate = a;
            bestSoFar[a] = firstAtt;
            available.insert(a);
        }

        while (!available.empty())
        {
            const CategoricalAttribute current = topCandidate;
            parrents_local_y[local_y][current] = bestSoFar[current];
            available.erase(current);

            if (!available.empty())
            {
                topCandidate = *available.begin();
                for (std::set<CategoricalAttribute>::const_iterator it =
                        available.begin(); it != available.end(); it++)
                {
                    if (maxWeight[*it] < Iy_local[current][*it])
                    {
                        maxWeight[*it] = Iy_local[current][*it];
                        bestSoFar[*it] = current;
                    }

                    if (maxWeight[*it] > maxWeight[topCandidate])
                        topCandidate = *it;
                }
            }
        }
        delete[] bestSoFar;
        delete[] maxWeight;

        std::vector<double> classDist_local;
        classDist_local.resize(this->noClasses_);
        classify_local(inst,parrents_local_y[local_y],classDist_local);
        for(int _y = 0;_y<this->noClasses_;_y++){
            classDist_locals[_y] += classDist_local[_y];
            std::vector<double> classDist_gen;
            classDist_gen.resize(this->noClasses_);
            classify_local(inst,parents_y[_y],classDist_gen);
            double likehood = this->Cross_Entropy(classDist_local,classDist_gen);
            if(likehood < min_cross_entropy){
                model_selected = _y;
                min_cross_entropy = likehood;
            }
        }

        //printf("%d th models LikeHood: %lf\n",local_y,likehood);
    }
    std::vector<double> classDist_gen;
    classDist_gen.resize(this->noClasses_);
    classify_local(inst,parents_y[model_selected],classDist_gen);
    normalise(classDist_locals);
    for(int i = 0;i< this->noClasses_;i++){
        classDist[i] = classDist_gen[i] + classDist_locals[i];
    }
    normalise(classDist);


}

void TAN_IM6::finalisePass()
{
    assert(trainingIsFinished_ == false);

    //// calculate conditional mutual information
    //float **mi = new float *[meta->noAttributes];

    //for (attribute a = 0; a < meta->noAttributes; a++) {
    //  mi[a] = new float[meta->noAttributes];
    //}

    //const double totalCount = dist->xyCounts.count;

    //for (attribute x1 = 1; x1 < meta->noAttributes; x1++) {
    //  if (meta->attTypes[x1] == categorical) {
    //    for (attribute x2 = 0; x2 < x1; x2++) {
    //      if (meta->attTypes[x2] == categorical) {
    //        float m = 0.0;

    //        for (cat_value v1 = 0; v1 < meta->noValues[x1]; v1++) {
    //          for (cat_value v2 = 0; v2 < meta->noValues[x2]; v2++) {
    //            for (unsigned int y = 0; y < meta->noClasses(); y++) {
    //              const double x1x2y = dist->getCount(x1, v1, x2, v2, y);
    //              if (x1x2y) {
    //                //const unsigned int yCount = dist->xyCounts.getClassCount(y);
    //                //const unsigned int  x1y = dist->xyCounts.getCount(x1, v1, y);
    //                //const unsigned int  x2y = dist->xyCounts.getCount(x2, v2, y);
    //                m += (x1x2y/totalCount) * log(dist->xyCounts.getClassCount(y) * x1x2y / (static_cast<double>(dist->xyCounts.getCount(x1, v1, y))*dist->xyCounts.getCount(x2, v2, y)));
    //                //assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
    //              }
    //            }
    //          }
    //        }

    //        assert(m >= -0.00000001); // CMI is always positive, but allow for some imprecision
    //        mi[x1][x2] = m;
    //        mi[x2][x1] = m;
    //      }
    //    }
    //  }
    //}

    //初始化
    for(int i =0;i<this->Iy.size();i++){
        this->Iy[i].resize(this->noCatAtts_);
        for(int j =0;j<this->Iy[i].size();j++){
            this->Iy[i][j].resize(this->noCatAtts_);
        }
    }
    // 建立 y 个模型
    for(int y = 0;y< this->noClasses_;y++){
        //获取 判断依据
        //参考公式: 2018070301
        for(int xi = 0;xi<this->noCatAtts_;xi++){
            for(int xj =0;xj<this->noCatAtts_;xj++){
                if(xi == xj){
                    this->Iy[y][xi][xj] = 10000000;
                    continue;
                }
                else{
                    this->Iy[y][xi][xj] = 0.0;
                    for(int xi_num = 0;xi_num<this->instanceStream_->getNoValues(xi);xi_num++){
                        for(int xj_num = 0;xj_num<this->instanceStream_->getNoValues(xj);xj_num++){
                            this->Iy[y][xi][xj] +=(
                                (this->xxyDist_.jointP(xi,xi_num,xj,xj_num,y) / this->xxyDist_.xyCounts.p(y))
                                                   *
                                log2(
                                     (this->xxyDist_.jointP(xi,xi_num,xj,xj_num,y) / this->xxyDist_.xyCounts.p(y))
                                                                   /
                                     (this->xxyDist_.xyCounts.p(xi,xi_num,y) * this->xxyDist_.xyCounts.p(xj,xj_num,y))
                                )
                            );
                        }
                    }
                }

            }
        }
        //建模
        CategoricalAttribute firstAtt = 0;
        parents_y[y][firstAtt] = NOPARENT;
        float *maxWeight;
        CategoricalAttribute *bestSoFar;
        CategoricalAttribute topCandidate = firstAtt;
        std::set<CategoricalAttribute> available;
        safeAlloc(maxWeight, noCatAtts_);
        safeAlloc(bestSoFar, noCatAtts_);
        maxWeight[firstAtt] = -std::numeric_limits<float>::max();

        for (CategoricalAttribute a = firstAtt + 1; a < noCatAtts_; a++)
        {
            maxWeight[a] = Iy[y][firstAtt][a];
            if (Iy[y][firstAtt][a] > maxWeight[topCandidate])
                topCandidate = a;
            bestSoFar[a] = firstAtt;
            available.insert(a);
        }

        while (!available.empty())
        {
            const CategoricalAttribute current = topCandidate;
            parents_y[y][current] = bestSoFar[current];
            available.erase(current);

            if (!available.empty())
            {
                topCandidate = *available.begin();
                for (std::set<CategoricalAttribute>::const_iterator it =
                        available.begin(); it != available.end(); it++)
                {
                    if (maxWeight[*it] < Iy[y][current][*it])
                    {
                        maxWeight[*it] = Iy[y][current][*it];
                        bestSoFar[*it] = current;
                    }

                    if (maxWeight[*it] > maxWeight[topCandidate])
                        topCandidate = *it;
                }
            }
        }
       delete[] bestSoFar;
       delete[] maxWeight;
    }





    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool TAN_IM6::trainingIsFinished()
{
    return trainingIsFinished_;
}


