/* Open source system for classification learning from very large data
 ** Copyright (C) 2012 Geoffrey I Webb
 ** Implements Sahami's k-dependence Bayesian classifier
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
#include <assert.h>
#include <math.h>
#include <set>
#include <algorithm>
#include <stdlib.h>

#include "kdb_im2.h"
#include "utils.h"
#include "correlationMeasures.h"
#include "globals.h"

KDB_IM2::KDB_IM2() : pass_(1) //？？
{
}

KDB_IM2::KDB_IM2(char*const*& argv, char*const* end) : pass_(1)
{
    name_ = "KDB";

    // defaults
    k_ = 1;

    // get arguments
    while (argv != end)
    {
        if (*argv[0] != '-')
        {
            break;
        } else if (argv[0][1] == 'k')
        {
            getUIntFromStr(argv[0] + 2, k_, "k");
        } else
        {
            break;
        }

        name_ += argv[0];

        ++argv;
    }
}

KDB_IM2::~KDB_IM2(void)
{
}

void KDB_IM2::getCapabilities(capabilities &c)
{
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

// creates a comparator for two attributes based on their relative mutual information with the class

class miCmpClass
{
public:

    miCmpClass(std::vector<double> *m)
    {
        mi = m;
    }

    bool operator()(CategoricalAttribute a, CategoricalAttribute b)
    {
        return (*mi)[a] > (*mi)[b];
    }

private:
    std::vector<double> *mi;
};
//计算两个属性变量与类变量结点C之间的互信息，返回的是互信息最大的那个属性结点

void KDB_IM2::reset(InstanceStream &is)
{
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();

    k_ = min(k_, noCatAtts_ - 1); // k cannot exceed the real number of categorical attributes - 1
    //K_表示属性结点可以作为父节点的结点的个数
    // initialise distributions
    dTree_gens.resize(this->noClasses_);
    parents_gens.resize(this->noClasses_);
    for(int i =0;i<this->noClasses_;i++){
        dTree_gens[i].resize(noCatAtts);
        parents_gens[i].resize(noCatAtts);


        for (CategoricalAttribute a = 0; a < noCatAtts; a++)
        {
            parents_gens[i][a].clear(); //？？
            dTree_gens[i][a].init(is, a); //DTree   used in the second pass and for classification
        }

    }

    /*初始化各数据结构空间*/
    dist_.reset(is); //xxyDist dist_;
    xxxyDist_.reset(is);

    classDist_.reset(is); //yDist classDist_;

    pass_ = 1;
}
void KDB_IM2::classify_local(const instance& inst, std::vector<distributionTree> & dTree_,std::vector<double> &posteriorDist)
{
    // calculate the class probabilities in parallel
    // P(y)
    for (CatValue y = 0; y < noClasses_; y++)
    {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
    }

    // P(x_i | x_p1, .. x_pk, y)
    for (CategoricalAttribute x = 0; x < noCatAtts_; x++)
    {

        dTree_[x].updateClassDistribution(posteriorDist, x, inst);
    }

    // normalise the results
    normalise(posteriorDist);
}

void KDB_IM2::classify_local(const instance& inst,std::vector<std::vector<CategoricalAttribute> > parrent,std::vector<double> &posteriorDist)
{
    // calculate the class probabilities in parallel
    // P(y)
    for (CatValue y = 0; y < noClasses_; y++)
    {
        posteriorDist[y] = classDist_.p(y) * (std::numeric_limits<double>::max() / 2.0); // scale up by maximum possible factor to reduce risk of numeric underflow
    }


    for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
        // const CategoricalAttribute parent1 = parents_1[x1][0];//typedef CategoricalAttribute parent unsigned int
        // const CategoricalAttribute parent2 = parents_1[x1][1];

        if (parrent[x1].size()==0) {
            // printf("PARent=0  \n");
            for (CatValue y = 0; y < noClasses_; y++) {
                posteriorDist[y] *=xxxyDist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y);  // p(a=v|Y=y) using M-estimate
                //  printf("x1=%d     y=%d\n",x1,y);
                //  printf("PARent=0  :%f\n",dist_.xxyCounts.xyCounts.p(x1, inst.getCatVal(x1), y));
            }
        }
        else if(parrent[x1].size()==1){
            //  printf("PARent=1  \n");
            for (CatValue y = 0; y < noClasses_; y++) {
                posteriorDist[y] *=xxxyDist_.xxyCounts.p(x1, inst.getCatVal(x1), parrent[x1][0],inst.getCatVal(parrent[x1][0]), y); // p(x1=v1|Y=y, x2=v2) using M-estimate
                //classDist_2[y] *=dist_.xxyCounts.p(x1, inst.getCatVal(x1), parent12,inst.getCatVal(parent12), y);
                //printf("x1=%d       parents_1[x1][0]=%d     y=%d\n",x1,parents_1[x1][0],y);
                //    printf("PARent=1  :%f\n",dist_.xxyCounts.p(x1, inst.getCatVal(x1), parents_[x1][0],inst.getCatVal(parents_[x1][0]), y));
            }
        }
        else if(parrent[x1].size()==2){
            //  printf("PARent=2  \n");
            for (CatValue y = 0; y < noClasses_; y++) {// p(x1=v1|Y=y, x2=v2, x3=v3) using M-estimate
               posteriorDist[y] *= xxxyDist_.p(x1, inst.getCatVal(x1), parrent[x1][0],inst.getCatVal(parrent[x1][0]),parrent[x1][1],inst.getCatVal(parrent[x1][1]), y);
               // classDist_2[y] *= dist_.p(x1, inst.getCatVal(x1), parent12,inst.getCatVal(parent12),parent22,inst.getCatVal(parent22), y);
               // printf("x1=%d       parents_1[x1][0]=%d     parents_1[x1][1]=%d     y=%d\n",x1,parents_1[x1][0],parents_1[x1][1],y);
               //    printf("PARent=2  :%f\n",dist_.p(x1, inst.getCatVal(x1), parents_[x1][0],inst.getCatVal(parents_[x1][0]),parents_[x1][1],inst.getCatVal(parents_[x1][1]), y));
            }
        }
    }

    // normalise the results
    normalise(posteriorDist);
}

double KDB_IM2:: Cross_Entropy(std::vector<double> cd_1ocal,std::vector<double> cd_gen){
    double result = 0.0;
    for(int y = 0;y<this->noClasses_;y++){
        result+=(
            - cd_gen[y] * log2(cd_1ocal[y])
        );
    }
    return result;
}

/// primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass

/*通过训练集来填写数据空间*/
void KDB_IM2::train(const instance &inst)
{
    if (pass_ == 1)
    {
        dist_.update(inst);//只更新xxyDist
        xxxyDist_.update(inst);
    }
    else
    {
        //printf("pass_==2!!");
        assert(pass_ == 2);
        for(int i = 0;i<this ->noClasses_;i++){
            for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
            {
                dTree_gens[i][a].update(inst, a, parents_gens[i][a]);
            }
        }


        classDist_.update(inst);
    }
}

/// must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)

void KDB_IM2::initialisePass()
{
}

/// must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)

void KDB_IM2::finalisePass()
{
    if (pass_ == 1 && k_ != 0)
    {
        for(int y = 0;y<this->noClasses_;y++){
            std::vector<double> mi;
            mi.resize(this->noCatAtts_);
            //getMutualInformation(dist_.xyCounts, mi);//计算好了I(Xi;C)
            //计算 mi
            for(int i = 0;i<this->noCatAtts_;i++){
                mi[i] = 0.0;
                for(int i_num = 0;i_num <this->dist_.getNoValues(i);i_num++){
                    mi[i]+=(
                        this->dist_.xyCounts.jointP(i,i_num,y)
                                *
                        log2(
                            this->dist_.xyCounts.jointP(i,i_num,y)
                                         /
                            (this->dist_.xyCounts.p(i,i_num) * this->dist_.xyCounts.p(y))
                        )

                    );
                }
            }
            //printf("after mi");
            // 计算cmi

            // calculate the conditional mutual information from the xxy distribution
            //crosstab<float> cmi = crosstab<float>(noCatAtts_);
            //初始化
            std::vector<std::vector<double> >  Iy;

            Iy.resize(this->noCatAtts_);
            for(int j =0;j<Iy.size();j++){
                Iy[j].resize(this->noCatAtts_);
            }
            for(int xi = 0;xi<this->noCatAtts_;xi++){
                for(int xj =0;xj<this->noCatAtts_;xj++){
                    if(xi == xj){
                        Iy[xi][xj] = 10000000;
                        continue;
                    }
                    else{
                        Iy[xi][xj] = 0.0;
                        for(int xi_num = 0;xi_num<this->instanceStream_->getNoValues(xi);xi_num++){
                            for(int xj_num = 0;xj_num<this->instanceStream_->getNoValues(xj);xj_num++){
                                Iy[xi][xj] +=(
                                    (this->dist_.jointP(xi,xi_num,xj,xj_num,y) / this->dist_.xyCounts.p(y))
                                                   *
                                    log2(
                                         (this->dist_.jointP(xi,xi_num,xj,xj_num,y) / this->dist_.xyCounts.p(y))
                                                                   /
                                         (this->dist_.xyCounts.p(xi,xi_num,y) * this->dist_.xyCounts.p(xj,xj_num,y))
                                    )
                                );
                            }
                        }
                    }

                }
            }
            // calculate the mutual information from the xy distribution



            // sort the attributes on MI with the class
            std::vector<CategoricalAttribute> order; //？？order存放的是所有的属性结点

            for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
            {
                order.push_back(a);
            }

            // assign the parents
            if (!order.empty())
            {
                miCmpClass cmp(&mi);

                std::sort(order.begin(), order.end(), cmp); //？？

                // proper KDB assignment of parents
                for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++)
                {
                    parents_gens[y][*it].push_back(order[0]);
                    for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++)
                    {
                        // make parents into the top k attributes on mi that precede *it in order
                        if (parents_gens[y][*it].size() < k_)
                        {
                            // create space for another parent
                            // set it initially to the new parent.
                            // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                            parents_gens[y][*it].push_back(*it2);
                        }
                        for (unsigned int i = 0; i < parents_gens[y][*it].size(); i++)
                        {
                            if (Iy[*it2][*it] > Iy[parents_gens[y][*it][i]][*it])
                            {
                                // move lower value parents down in order
                                for (unsigned int j = parents_gens[y][*it].size() - 1; j > i; j--)
                                {
                                   parents_gens[y][*it][j] = parents_gens[y][*it][j - 1];
                                }
                                // insert the new att
                                parents_gens[y][*it][i] = *it2;
                                break;
                           }
                       }
                    }
                }
            }

        }

    }

    ++pass_;
}

// true if no more passes are required. updated by finalisePass()

bool KDB_IM2::trainingIsFinished()
{
    return pass_ > 2;
}

void KDB_IM2::classify(const instance& inst, std::vector<double> &posteriorDist)
{
    std::vector< std::vector<std::vector<CategoricalAttribute> > > parents_locals;
    parents_locals.resize(this->noClasses_);
    for(int i =0;i<this->noClasses_;i++){
        parents_locals[i].resize(this->noCatAtts_);

        for (CategoricalAttribute a = 0; a < this->noCatAtts_; a++)
        {
            parents_locals[i][a].clear(); //？？
        }

    }

    for(int y = 0;y<this->noClasses_;y++){
        std::vector<double> mi;
        mi.resize(this->noCatAtts_);
        for(int i = 0;i<this->noCatAtts_;i++){
            //mi[i] = 0.0;
            mi[i]=(
                this->dist_.xyCounts.jointP(i,inst.getCatVal(i),y)
                                *
                log2(
                    this->dist_.xyCounts.jointP(i,inst.getCatVal(i),y)
                                        /
                    (this->dist_.xyCounts.p(i,inst.getCatVal(i)) * this->dist_.xyCounts.p(y))
                )

            );
        }

        std::vector<std::vector<double> >  Iy;

        Iy.resize(this->noCatAtts_);
        for(int j =0;j<Iy.size();j++){
            Iy[j].resize(this->noCatAtts_);
        }
        for(int xi = 0;xi<this->noCatAtts_;xi++){
            for(int xj =0;xj<this->noCatAtts_;xj++){
                if(xi == xj){
                    Iy[xi][xj] = 10000000;
                    continue;
                }
                else{
                    Iy[xi][xj] = 0.0;
                    Iy[xi][xj] +=(
                        (this->dist_.jointP(xi,inst.getCatVal(xi),xj,inst.getCatVal(xj),y) / this->dist_.xyCounts.p(y))
                                                   *
                        log2(
                            (this->dist_.jointP(xi,inst.getCatVal(xi),xj,inst.getCatVal(xj),y) / this->dist_.xyCounts.p(y))
                                                                   /
                            (this->dist_.xyCounts.p(xi,inst.getCatVal(xi),y) * this->dist_.xyCounts.p(xj,inst.getCatVal(xj),y))
                        )
                    );
                }
            }
        }
        // sort the attributes on MI with the class
        std::vector<CategoricalAttribute> order; //？？order存放的是所有的属性结点

        for (CategoricalAttribute a = 0; a < noCatAtts_; a++)
        {
            order.push_back(a);
        }

        // assign the parents
        if (!order.empty())
        {
            miCmpClass cmp(&mi);

            std::sort(order.begin(), order.end(), cmp); //？？

                // proper KDB assignment of parents
            for (std::vector<CategoricalAttribute>::const_iterator it = order.begin() + 1; it != order.end(); it++)
            {
                parents_locals[y][*it].push_back(order[0]);
                for (std::vector<CategoricalAttribute>::const_iterator it2 = order.begin() + 1; it2 != it; it2++)
                {
                    // make parents into the top k attributes on mi that precede *it in order
                    if (parents_locals[y][*it].size() < k_)
                    {
                        // create space for another parent
                        // set it initially to the new parent.
                        // if there is a lower value parent, the new parent will be inserted earlier and this value will get overwritten
                        parents_locals[y][*it].push_back(*it2);
                    }
                    for (unsigned int i = 0; i < parents_locals[y][*it].size(); i++)
                    {
                        if (Iy[*it2][*it] > Iy[parents_locals[y][*it][i]][*it])
                        {
                            // move lower value parents down in order
                            for (unsigned int j = parents_locals[y][*it].size() - 1; j > i; j--)
                            {
                                parents_locals[y][*it][j] = parents_locals[y][*it][j - 1];
                            }
                            // insert the new att
                            parents_locals[y][*it][i] = *it2;
                            break;
                        }
                    }
                }
            }
        }

    }
    std::vector<std::vector<double> > classDict_gens;
    std::vector<std::vector<double> > classDict_locals;
    classDict_gens.resize(this->noClasses_);
    classDict_locals.resize(this->noClasses_);
    for(int y=0;y<this->noClasses_;y++){
        classDict_gens[y].resize(this->noClasses_);
        classDict_locals[y].resize(this->noClasses_);
        this->classify_local(inst,dTree_gens[y],classDict_gens[y]);
        this->classify_local(inst,parents_locals[y],classDict_locals[y]);
    }
    // 个性化定制
    int model_select=0;
    double min_value =1000000000;
    for(int y1 =0;y1<this->noClasses_;y1++){
        double likehood = this->Cross_Entropy(classDict_locals[y1],classDict_gens[y1]);
        if(likehood < min_value){
            min_value = likehood;
            model_select = y1;
        }
    }
    for(int y1 =0;y1<this->noClasses_;y1++){
        posteriorDist[y1] = classDict_gens[model_select][y1];
    }
}



