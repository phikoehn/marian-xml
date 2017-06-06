#pragma once

#include <vector>
#include <boost/iterator/permutation_iterator.hpp>

#include "common/scorer.h"
#include "common/god.h"
#include "common/exception.h"
#include "cpu/mblas/matrix.h"

namespace amunmt {
namespace CPU {

struct ProbCompare {
  ProbCompare(const float* data) : data_(data) {}

  bool operator()(const unsigned a, const unsigned b) {
    return data_[a] > data_[b];
  }

  const float* data_;
};

class BestHyps : public BestHypsBase
{
  public:
    BestHyps(const God &god)
      : BestHypsBase(
          !god.Get<bool>("allow-unk"),
          god.Get<bool>("n-best"),
          god.Get<std::vector<std::string>>("softmax-filter").size(),
          god.Get<bool>("return-alignment") || god.Get<bool>("return-soft-alignment"),
          god.GetScorerWeights())
    {}

    void CalcBeam(
        const God &god,
        const Beam& prevHyps,
        const std::vector<ScorerPtr>& scorers,
        const Words& filterIndices,
        std::vector<Beam>& beams,
        std::vector<size_t>& beamSizes)
    {
      using namespace mblas;

      mblas::ArrayMatrix& Probs = static_cast<mblas::ArrayMatrix&>(scorers[0]->GetProbs());

      mblas::ArrayMatrix Costs(Probs.rows(), 1);
      for (size_t i = 0; i < prevHyps.size(); ++i) {
        Costs.data()[i] = prevHyps[i]->GetCost();
      }

      Probs *= weights_.at(scorers[0]->GetName());
      AddBiasVector<byColumn>(Probs, Costs);

      for (size_t i = 1; i < scorers.size(); ++i) {
        mblas::ArrayMatrix &currProb = static_cast<mblas::ArrayMatrix&>(scorers[i]->GetProbs());

        Probs += weights_.at(scorers[i]->GetName()) * currProb;
      }

    std::vector< std::vector<XmlOptionCovered> > xmlCovered;
    for (size_t i = 0; i < prevHyps.size(); ++i) {
      xmlCovered.push_back( prevHyps[i]->GetXmlOptionCovered() );
    }

    std::cerr << "prevHyps.size() = " << prevHyps.size() << "\n";
    std::vector<SoftAlignmentPtr> alignments;
    if (CPU::EncoderDecoder* encdec = dynamic_cast<CPU::EncoderDecoder*>(scorers[0].get())) {
      auto& attention = encdec->GetAttention();
      std::cerr << "attention.rows() = " << attention.rows() << "\n";
      for(size_t i = 0; i < attention.rows(); i++) {
        alignments.emplace_back(new SoftAlignment(attention.begin(i),
                                                  attention.end(i)));
      }
    }
    else {
      std::cerr << "FAILED";
    }

    //std::cerr << "Probs.rows() = " << Probs.rows() << "\n";
    //std::cerr << "Probs.columns() = " << Probs.columns() << "\n";
    //std::cerr << "Probs.size() = " << Probs.size() << "\n";
    float  penaltyWeight = god.Get<float>("xml-penalty-weight");
    size_t penaltyWindow = god.Get<float>("xml-penalty-window");
    mblas::ArrayMatrix XmlCoveragePenalty(Probs.rows(), 1);

    std::cerr << "alignments for current hypotheses:\n";
    for (size_t i = 0; i < prevHyps.size(); ++i) {
      size_t max_pos = -1;
      float max_value = 0;
      SoftAlignment alignment = *(alignments[i]);
      float eos_attention = alignment[alignment.size()-1];
      for (size_t j = 0; j < alignment.size()-1; j++) {
        alignment[j] /= 1 - eos_attention;
        if (alignment[j] < 0.1) { std::cerr << "-"; }
        else { std::cerr << (int) (alignment[j]*10); }
        if (alignment[j] > max_value) {
          max_value = alignment[j];
          max_pos = j;
        }
      }
      //if (prevHyps[i]->GetLength() >= 1)
      //  std::cerr << " " << god.GetTargetVocab()[prevHyps[i]->GetPrevHyp()->GetWord()];
      //std::cerr << " " << god.GetTargetVocab()[prevHyps[i]->GetWord()];
      HypothesisPtr hyp = prevHyps[i];
      while(hyp->GetLength() > 0) {
        std::cerr << " " << god.GetTargetVocab()[hyp->GetWord()];
        hyp = hyp->GetPrevHyp();
      }
      //float prevCost = 0.0;
      //if (prevHyps[i]->GetLength() > 0) {
        float prevCost = prevHyps[i]->GetCost();
        //std::cerr << " prevCost:" << prevCost;
      //}

      bool coveredByXml = false;
      if (god.Get<bool>("xml-input") && xmlCovered[i].size()>0) {
        // is there an xmlOption that was started but is not complete?
        for(size_t j=0; j<xmlCovered[i].size(); j++) {
          if (xmlCovered[i][j].GetStarted() && !xmlCovered[i][j].GetCovered()) {
            std::cerr << " continue with XML option" ;
            const Words &outputWords = xmlCovered[i][j].GetOption()->GetOutput();
            Word outputWord = outputWords[xmlCovered[i][j].GetPosition()];
            auto ProbsPtx = Probs.begin();
            ProbsPtx += Probs.columns() * i;
            for(size_t k = 0; k < Probs.columns(); k++, ProbsPtx++) {
              if (k == outputWord) {
                *ProbsPtx = prevCost;
                // fourth fix below
                // *ProbsPtx = prevCost+1;
                // first fix below
                //if (*ProbsPtx < prevCost-5) {
                //  *ProbsPtx = prevCost-5;
                //}
                //*ProbsPtx = prevCost; // i.e. prob = 100% // TODO: MATCH BELOW
              }
              else {
                *ProbsPtx += -999.0;
              }
            }
            xmlCovered[i][j].Proceed();
            if (xmlCovered[i][j].GetCovered()) {
              std::cerr << ", now complete";
            }
            coveredByXml = true;
          }
        }

        // does a new xmlOption start here?
        for (size_t j=0; j<xmlCovered[i].size() && !coveredByXml; j++) {
          //std::cerr << max_pos << "=" << xmlOptions[j].GetStart() << "." << xmlCovered[i][j].GetCovered() << " ";
          if (max_pos == xmlCovered[i][j].GetOption()->GetStart() &&
              !xmlCovered[i][j].GetCovered()) {
            std::cerr << " XML" ;
            const Words &outputWords = xmlCovered[i][j].GetOption()->GetOutput();
            Word outputWord = outputWords[0];
            auto ProbsPtx = Probs.begin();
            ProbsPtx += Probs.columns() * i;
            for(size_t k = 0; k < Probs.columns(); k++, ProbsPtx++) {
              if (k == outputWord) {
                *ProbsPtx = prevCost;
                // fourth fix below
                // *ProbsPtx = prevCost+1;
                // first fix below
                // if (*ProbsPtx < prevCost-5) {
                //   *ProbsPtx = prevCost-5;
                // }
                // *ProbsPtx = prevCost; // i.e. prob = 100%
              }
              else {
                *ProbsPtx += -999.0;
              }
            }
            xmlCovered[i][j].Start();
            if (xmlCovered[i][j].GetCovered()) {
              std::cerr << ", complete" << outputWords.size();
            }
            else {
              std::cerr << ", tbc" << outputWords.size();
            }
            coveredByXml = true;
          }
        }
        // xml coverage penalties
        size_t translationLength = prevHyps[i]->GetLength();
        float penalty = 0.0;
        float allP = 0.0;
        std::cerr << ", penalty";
        for(size_t j=0; j<xmlCovered[i].size(); j++) {
          penalty -= prevHyps[i]->GetXmlOptionCovered()[j].GetPenalty(penaltyWeight, penaltyWindow, translationLength-1, false);
          penalty += xmlCovered[i][j].GetPenalty( penaltyWeight, penaltyWindow, translationLength, false );
          allP += xmlCovered[i][j].GetPenalty( penaltyWeight, penaltyWindow, translationLength, false );
          std::cerr << "," << j << "(" << penaltyWeight << "," << penaltyWindow << "," << translationLength << ")=" << xmlCovered[i][j].GetPenalty( penaltyWeight, penaltyWindow, translationLength, false ) << "/" << prevHyps[i]->GetXmlOptionCovered()[j].GetPenalty(penaltyWeight, penaltyWindow, translationLength-1, false);
        }
        XmlCoveragePenalty.data()[i] = penalty;
        std::cerr << " penalty:" << allP << " add" << penalty;
      }

      if (god.Has("lexicon-bias") && !coveredByXml) {
        auto ProbsPtx = Probs.begin() + Probs.columns() * i;
        for (auto& x: god.GetLexiconBias()) {
          size_t word = x.first;
          float bias = x.second;
          *(ProbsPtx + word) += bias;
        }
      }
      std::cerr << " cost:" << prevHyps[i]->GetCost();
      //std::cerr << " " << prevHyps[i]->GetCostBreakdown().size();
      std::vector<float>& breakdown = prevHyps[i]->GetCostBreakdown();
      if (breakdown.size() == 4) {
        std::cerr << " breakdown:" << breakdown[0] << " " << breakdown[1] << " " << breakdown[2] << " " << breakdown[3];
      }
      std::cerr << "\n";
    }


    if (god.Get<bool>("xml-input") && xmlCovered[0].size()>0) {
      std::cerr << Debug(XmlCoveragePenalty);
      AddBiasVector<byColumn>(Probs, XmlCoveragePenalty);
      for (size_t i = 0; i < prevHyps.size(); ++i) {
        size_t translationLength = prevHyps[i]->GetLength();
        // if predicted word is EOS, then apply final penalty
        for(size_t j=0; j<xmlCovered[i].size(); j++) {
          auto ProbsPtx = Probs.begin() + Probs.columns() * i + EOS_ID;
          *ProbsPtx -= xmlCovered[i][j].GetPenalty( penaltyWeight, penaltyWindow, translationLength, false );
          *ProbsPtx += xmlCovered[i][j].GetPenalty( penaltyWeight, penaltyWindow, translationLength, true );
        }
      }
    }



      size_t size = Probs.rows() * Probs.columns(); // Probs.size();
      std::vector<size_t> keys(size);
      for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = i;
      }

      size_t beamSize = beamSizes[0];

      std::vector<size_t> bestKeys(beamSize);
      std::vector<float> bestCosts(beamSize);

      if (forbidUNK_) {
        blaze::column(Probs, UNK_ID) = std::numeric_limits<float>::lowest();
      }

      std::nth_element(keys.begin(), keys.begin() + beamSize, keys.end(),
                       ProbCompare(Probs.data()));

      for (size_t i = 0; i < beamSize; ++i) {
        bestKeys[i] = keys[i];
        bestCosts[i] = Probs.data()[keys[i]];
      }

      std::vector<std::vector<float>> breakDowns;
      if (returnNBestList_) {
        breakDowns.push_back(bestCosts);
        for (auto& scorer : scorers) {
          std::vector<float> modelCosts(beamSize);
          mblas::ArrayMatrix &currProb = static_cast<mblas::ArrayMatrix&>(scorer->GetProbs());

          auto it = boost::make_permutation_iterator(currProb.begin(), keys.begin());
          std::copy(it, it + beamSize, modelCosts.begin());
          breakDowns.push_back(modelCosts);
        }
      }

      for (size_t i = 0; i < beamSize; i++) {
        size_t wordIndex = bestKeys[i] % Probs.columns();

        if (isInputFiltered_) {
          wordIndex = filterIndices[wordIndex];
        }

        size_t hypIndex  = bestKeys[i] / Probs.columns();
        float cost = bestCosts[i];

        HypothesisPtr hyp;
        if (returnAttentionWeights_) {
          std::vector<SoftAlignmentPtr> alignments;
          for (auto& scorer : scorers) {
            if (CPU::EncoderDecoder* encdec = dynamic_cast<CPU::EncoderDecoder*>(scorer.get())) {
              auto& attention = encdec->GetAttention();
              alignments.emplace_back(new SoftAlignment(attention.begin(hypIndex),
                                                        attention.end(hypIndex)));
            } else {
              amunmt_UTIL_THROW2("Return Alignment is allowed only with Nematus scorer.");
            }
          }

          hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost, xmlCovered[hypIndex], alignments));
        } else {
          hyp.reset(new Hypothesis(prevHyps[hypIndex], wordIndex, hypIndex, cost, xmlCovered[hypIndex]));
        }

        if (returnNBestList_) {
          hyp->GetCostBreakdown().resize(scorers.size());
          float sum = 0;
          for(size_t j = 0; j < scorers.size(); ++j) {
            if (j == 0) {
              hyp->GetCostBreakdown()[0] = breakDowns[0][i];
            } else {
              float cost = 0;
              if (j < scorers.size()) {
                if (prevHyps[hypIndex]->GetCostBreakdown().size() < scorers.size())
                  const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown().resize(scorers.size(), 0.0);
                cost = breakDowns[j][i] + const_cast<HypothesisPtr&>(prevHyps[hypIndex])->GetCostBreakdown()[j];
              }
              sum += weights_.at(scorers[j]->GetName()) * cost;
              hyp->GetCostBreakdown()[j] = cost;
            }
          }
          hyp->GetCostBreakdown()[0] -= sum;
          hyp->GetCostBreakdown()[0] /= weights_.at(scorers[0]->GetName());
        }
        beams[0].push_back(hyp);
      }
    }
};

}  // namespace CPU
}  // namespace amunmt
