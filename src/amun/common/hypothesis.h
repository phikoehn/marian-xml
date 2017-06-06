#pragma once
#include <memory>
#include "common/types.h"
#include "common/soft_alignment.h"
#include "common/sentence.h"

namespace amunmt {

class Hypothesis;

typedef std::shared_ptr<Hypothesis> HypothesisPtr;

class Hypothesis {
  public:
    Hypothesis()
     : prevHyp_(nullptr),
       prevIndex_(0),
       word_(0),
       length_(0),
       cost_(0.0)
    {}

    Hypothesis(const HypothesisPtr prevHyp, size_t word, size_t prevIndex, float cost, 
               std::vector<XmlOptionCovered> xmlCovered)
      : prevHyp_(prevHyp),
        prevIndex_(prevIndex),
        word_(word),
        length_(prevHyp->length_+1),
        xmlOptionCovered_(xmlCovered),
        cost_(cost)
    {}

    Hypothesis(const HypothesisPtr prevHyp, size_t word, size_t prevIndex, float cost,
               std::vector<XmlOptionCovered> xmlCovered,
               std::vector<SoftAlignmentPtr> alignment)
      : prevHyp_(prevHyp),
        prevIndex_(prevIndex),
        word_(word),
        cost_(cost),
        length_(prevHyp->length_+1),
        xmlOptionCovered_(xmlCovered),
        alignments_(alignment)
    {}

    const HypothesisPtr GetPrevHyp() const {
      return prevHyp_;
    }

    size_t GetWord() const {
      return word_;
    }

    size_t GetPrevStateIndex() const {
      return prevIndex_;
    }

    float GetCost() const {
      return cost_;
    }

    size_t GetLength() const {
      return length_;
    }

    std::vector<float>& GetCostBreakdown() {
      return costBreakdown_;
    }

    SoftAlignmentPtr GetAlignment(size_t i) {
      return alignments_[i];
    }

    std::vector<SoftAlignmentPtr>& GetAlignments() {
      return alignments_;
    }

    void InitXmlOptionCovered( const std::vector<XmlOption> &xmlOptionList ) {
      for(auto &xmlOption : xmlOptionList ) {
        XmlOptionCovered xmlCovered(&xmlOption);
        xmlOptionCovered_.push_back(xmlCovered);
      }
    }

    std::vector<XmlOptionCovered>& GetXmlOptionCovered() {
      return xmlOptionCovered_;
    }

  private:
    const HypothesisPtr prevHyp_;
    const size_t prevIndex_;
    const size_t word_;
    const float cost_;
    size_t length_;
    std::vector<SoftAlignmentPtr> alignments_;
    std::vector<XmlOptionCovered> xmlOptionCovered_;

    std::vector<float> costBreakdown_;
};

typedef std::vector<HypothesisPtr> Beam;
typedef std::vector<Beam> Beams;
typedef std::pair<Words, HypothesisPtr> Result;
typedef std::vector<Result> NBestList;

}

