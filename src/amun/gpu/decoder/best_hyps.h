#pragma once

#include <map>
#include <numeric>
#include <boost/timer/timer.hpp>

#include "common/scorer.h"
#include "common/exception.h"
#include "common/god.h"
#include "common/utils.h"
#include "gpu/mblas/matrix_functions.h"
#include "gpu/mblas/nth_element.h"

#include "gpu/decoder/encoder_decoder.h"

namespace amunmt {
namespace GPU {

class BestHyps : public BestHypsBase
{
  public:
    BestHyps(const BestHyps &copy) = delete;

    BestHyps(const God &god)
          : BestHypsBase(
              !god.Get<bool>("allow-unk"),
              god.Get<bool>("n-best"),
              god.Get<std::vector<std::string>>("softmax-filter").size(),
              god.Get<bool>("return-alignment") || god.Get<bool>("return-soft-alignment"),
              god.GetScorerWeights()),
            nthElement_(god.Get<size_t>("beam-size"), god.Get<size_t>("mini-batch")),
            keys(god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch")),
            Costs(god.Get<size_t>("beam-size") * god.Get<size_t>("mini-batch"))
    {}

    void DisAllowUNK(mblas::Matrix& Prob) {
      SetColumn(Prob, UNK_ID, std::numeric_limits<float>::lowest());
    }

    void FindBests(const std::vector<uint>& beamSizes, mblas::Matrix& Probs,
                   std::vector<float>& outCosts,
                   std::vector<unsigned>& outKeys,
                   const bool isFirst) {
      nthElement_.getNBestList(beamSizes, Probs, outCosts, outKeys, isFirst);
    }

    std::vector<SoftAlignmentPtr> GetAlignments(const std::vector<ScorerPtr>& scorers,
                                                size_t hypIndex) {
      std::vector<SoftAlignmentPtr> alignments;
      for (auto& scorer : scorers) {
        if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(scorer.get())) {
          const mblas::Matrix &attention = encdec->GetAttention();
          size_t attLength = attention.dim(1);

          SoftAlignment *softAlignment = new SoftAlignment(attLength);
          mblas::copy(
              attention.data() + hypIndex * attLength,
              attLength,
              thrust::raw_pointer_cast(softAlignment->data()),
              cudaMemcpyDeviceToHost
          );

          alignments.emplace_back(softAlignment);
        } else {
          amunmt_UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
        }
      }
      return alignments;
    }

    void CalcBeam(
        const God &god,
        const Beam& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        std::vector<Beam>& beams,
        std::vector<uint>& beamSizes)
    {
      BEGIN_TIMER("CalcBeam");

      using namespace mblas;
      bool debug = false;

      if (debug) std::cerr << "CalcBeam\n";
      mblas::Matrix& Probs = static_cast<mblas::Matrix&>(scorers[0]->GetProbs());

      HostVector<float> vCosts;
      for (auto& h : prevHyps) {
        vCosts.push_back(h->GetCost());
      }
      mblas::copy(vCosts.begin(), vCosts.end(), Costs.begin());

      const bool isFirst = (vCosts[0] == 0.0f) ? true : false;

      BroadcastVecColumn(weights_.at(scorers[0]->GetName()) * _1 + _2, Probs, Costs);

      for (size_t i = 1; i < scorers.size(); ++i) {
        mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

        Element(_1 + weights_.at(scorers[i]->GetName()) * _2, Probs, currProbs);
      }

      std::vector< std::vector<XmlOptionCovered> > xmlCovered;
      for (auto& h : prevHyps) {
        xmlCovered.push_back( h->GetXmlOptionCovered() );
      }

      std::vector<SoftAlignmentPtr> alignments;
      for (size_t i = 0; i < prevHyps.size(); ++i) {
        std::vector<SoftAlignmentPtr> alignmentForOneHyp = GetAlignments( scorers, i );
        alignments.push_back( alignmentForOneHyp[0] );
        //if (GPU::EncoderDecoder* encdec = dynamic_cast<GPU::EncoderDecoder*>(scorers[0].get())) {
        //  auto& attention = encdec->GetAttention();
        //  size_t attLength = attention.Cols();
        //  alignments.emplace_back(new SoftAlignment(
        //      attention.begin() + i * attLength,
        //      attention.begin() + (i + 1) * attLength));
        //}
        //else {
        //  std::cerr << "FAILED";
        //}
      }

      float  penaltyWeight = god.Get<float>("xml-penalty-weight");
      size_t penaltyWindow = god.Get<float>("xml-penalty-window");
      HostVector<float> vXmlCoveragePenalty;
      for (auto& h : prevHyps) {
        vXmlCoveragePenalty.push_back( 0.0 );
      };
      if (debug) std::cerr << "alignments for current hypotheses:\n";
      for (size_t i = 0; i < prevHyps.size(); ++i) {
        int max_pos = -1;
        float max_value = 0;
        SoftAlignment alignment = *(alignments[i]);
        float eos_attention = alignment[alignment.size()-1];
        for (size_t j = 0; j < alignment.size()-1; j++) {
          alignment[j] /= 1 - eos_attention;
          if (debug) {
            if (alignment[j] < 0.1) { std::cerr << "-"; }
            else { std::cerr << (int) (alignment[j]*10); }
          }
          if (alignment[j] > max_value) {
            max_value = alignment[j];
            max_pos = j;
          }
        }

        if (debug) {
          HypothesisPtr hyp = prevHyps[i];
          while(hyp->GetLength() > 0) {
            std::cerr << " " << god.GetTargetVocab()[hyp->GetWord()];
            hyp = hyp->GetPrevHyp();
          }
        }
        float prevCost = prevHyps[i]->GetCost();

        bool coveredByXml = false;
        if (god.Get<bool>("xml-input") && xmlCovered[i].size()>0) {

          // is there an xmlOption that was started but is not complete?
          for(size_t j=0; j<xmlCovered[i].size(); j++) {
            if (xmlCovered[i][j].GetStarted() && !xmlCovered[i][j].GetCovered()) {
              if (debug) std::cerr << " continue with XML option" ;
              const Words &outputWords = xmlCovered[i][j].GetOption()->GetOutput();
              Word outputWord = outputWords[xmlCovered[i][j].GetPosition()];
              for(size_t k = 0; k < Probs.dim(1); k++) {
                if (k == outputWord) {
                  Probs.set(prevCost, i, k, 0, 0);
                }
                else {
                  float val = Probs.get(k, i, 0, 0);
                  Probs.set(val-999, i, k, 0, 0);
                }
              }
              xmlCovered[i][j].Proceed();
              if (debug && xmlCovered[i][j].GetCovered()) {
                std::cerr << ", now complete";
              }
              coveredByXml = true;
            }
          }

          // does a new xmlOption start here?
          for (size_t j=0; j<xmlCovered[i].size() && !coveredByXml; j++) {
            if (max_pos == xmlCovered[i][j].GetOption()->GetStart() &&
                !xmlCovered[i][j].GetCovered()) {
              if (debug) std::cerr << " XML" ;
              const Words &outputWords = xmlCovered[i][j].GetOption()->GetOutput();
              Word outputWord = outputWords[0];
              Probs.set(999.0+prevCost, i, outputWord, 0, 0);
              vXmlCoveragePenalty[i] += -999.0;
              xmlCovered[i][j].Start();
              if (debug) {
                if (xmlCovered[i][j].GetCovered()) {
                  std::cerr << ", complete" << outputWords.size();
                }
                else {
                  std::cerr << ", tbc" << outputWords.size();
                }
              }
              coveredByXml = true;
            }
          }
          // xml coverage penalties
          size_t translationLength = prevHyps[i]->GetLength();
          float penalty = 0.0;
          float allP = 0.0;
          if (debug) std::cerr << ", penalty";
          for(size_t j=0; j<xmlCovered[i].size(); j++) {
            penalty -= prevHyps[i]->GetXmlOptionCovered()[j].GetPenalty(penaltyWeight, penaltyWindow, translationLength-1, false);
            penalty += xmlCovered[i][j].GetPenalty( penaltyWeight, penaltyWindow, translationLength, false );
            if (debug) allP += xmlCovered[i][j].GetPenalty( penaltyWeight, penaltyWindow, translationLength, false );
            if (debug) std::cerr << "," << j << "(" << penaltyWeight << "," << penaltyWindow << "," << translationLength << ")=" << xmlCovered[i][j].GetPenalty( penaltyWeight, penaltyWindow, translationLength, false ) << "/" << prevHyps[i]->GetXmlOptionCovered()[j].GetPenalty(penaltyWeight, penaltyWindow, translationLength-1, false);
          }
          vXmlCoveragePenalty[i] += penalty;
          if (debug) std::cerr << " penalty:" << allP << " add" << penalty;

        }

        if (god.Has("lexicon-bias") && !coveredByXml) {
          for (auto& x: god.GetLexiconBias()) {
            size_t word = x.first;
            float bias = x.second;
            float val = Probs.get(i, word, 0, 0);
            Probs.set(val + bias, i, word, 0, 0);
          }
        }
        if (debug) std::cerr << " cost:" << prevHyps[i]->GetCost();
        if (debug) std::cerr << "\n";
      }

      if (god.Get<bool>("xml-input") && xmlCovered[0].size()>0) {
        DeviceVector<float> XmlCoveragePenalty(vXmlCoveragePenalty.size());;
        mblas::copy(vXmlCoveragePenalty.begin(), vXmlCoveragePenalty.end(), XmlCoveragePenalty.begin());
        BroadcastVecColumn(_1 + _2, Probs, XmlCoveragePenalty);

        // std::cerr << Debug(XmlCoveragePenalty);
        // TODO AddBiasVector<byColumn>(Probs, XmlCoveragePenalty);
        for (size_t i = 0; i < prevHyps.size(); ++i) {
          size_t translationLength = prevHyps[i]->GetLength();
          // if predicted word is EOS, then apply final penalty
          for(size_t j=0; j<xmlCovered[i].size(); j++) {
            float val = Probs.get(i, EOS_ID, 0, 0);
            val -= xmlCovered[i][j].GetPenalty( penaltyWeight, penaltyWindow, translationLength, false );
            val += xmlCovered[i][j].GetPenalty( penaltyWeight, penaltyWindow, translationLength, true );
            Probs.set(val, i, EOS_ID, 0, 0);
          }
        }
      }

      if (forbidUNK_) {
        DisAllowUNK(Probs);
      }

      size_t beamSizeSum = std::accumulate(beamSizes.begin(), beamSizes.end(), 0);

      std::vector<float> bestCosts;
      std::vector<unsigned> bestKeys;

      FindBests(beamSizes, Probs, bestCosts, bestKeys, isFirst);

      std::vector<HostVector<float>> breakDowns;
      if (returnNBestList_) {
          breakDowns.push_back(bestCosts);
          for (size_t i = 1; i < scorers.size(); ++i) {
            std::vector<float> modelCosts(beamSizeSum);
            mblas::Matrix &currProbs = static_cast<mblas::Matrix&>(scorers[i]->GetProbs());

            nthElement_.getValueByKey(modelCosts, currProbs);
            breakDowns.push_back(modelCosts);
          }
      }

      std::map<size_t, size_t> batchMap;
      size_t tmp = 0;
      for (size_t batchID = 0; batchID < beamSizes.size(); ++batchID) {
        for (size_t t = 0; t < beamSizes[batchID]; ++t) {
          batchMap[tmp++] = batchID;
        }
      }

      for (size_t i = 0; i < beamSizeSum; i++) {
        size_t wordIndex = bestKeys[i] % Probs.dim(1);
        if (isInputFiltered_) {
          wordIndex = filterIndices[wordIndex];
        }

        size_t hypIndex  = bestKeys[i] / Probs.dim(1);
        float cost = bestCosts[i];

        HypothesisPtr hyp;
        if (returnAttentionWeights_) {
          hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost,
                                   xmlCovered[hypIndex],
                                   GetAlignments(scorers, hypIndex)));
        } else {
          hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost,
                                   xmlCovered[hypIndex]));
        }

        if(returnNBestList_) {
          hyp->GetCostBreakdown().resize(scorers.size());
          float sum = 0;
          for (size_t j = 0; j < scorers.size(); ++j) {
            if (j == 0)
              hyp->GetCostBreakdown()[0] = breakDowns[0][i];
            else {
              float cost = 0;
              if (j < scorers.size()) {
                  if (prevHyps[hypIndex]->GetCostBreakdown().size() < scorers.size())
                    const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(scorers.size(), 0.0f);
                  cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
              }
              sum += weights_.at(scorers[j]->GetName()) * cost;
              hyp->GetCostBreakdown()[j] = cost;
            }
          }
          hyp->GetCostBreakdown()[0] -= sum;
          hyp->GetCostBreakdown()[0] /= weights_.at(scorers[0]->GetName());
        }

        beams[batchMap[i]].push_back(hyp);
      }

      PAUSE_TIMER("CalcBeam");
    }


  private:
    NthElement nthElement_;
    DeviceVector<unsigned> keys;
    DeviceVector<float> Costs;
};

}
}

