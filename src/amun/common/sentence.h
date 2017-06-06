#pragma once
#include <memory>
#include <vector>
#include <string>
#include "types.h"

namespace amunmt {

class God;

class XmlOption {
  public:
    XmlOption(size_t start, size_t end, Words output)
      : start_(start),
        end_(end),
       output_(output) {
    }

    size_t GetStart() const {
      return start_;
    }

    size_t GetEnd() const {
      return end_;
    }

    const Words GetOutput() const {
      return output_;
    }

  private:
    size_t start_;
    size_t end_;
    Words output_;
};

class XmlOptionCovered {
  public:
    XmlOptionCovered(const XmlOption *option)
      : started_(false),
        covered_(false),
        option_(option) {
    }

    bool GetStarted() {
      return started_;
    }

    bool GetCovered() {
      return covered_;
    }

    bool GetPosition() {
      return position_;
    }

    const XmlOption* GetOption() {
      return option_;
    }

    void Start() {
      started_ = true;
      position_ = 1;
      if (option_->GetOutput().size() == 1) {
        covered_ = true;
      }
      // std::cerr << "option" << option_->GetOutput().size();
    }
    void Proceed() {
      position_++;
      if (option_->GetOutput().size() == position_) {
        covered_ = true;
      }
    }

    float GetPenalty( float penaltyWeight, size_t windowSize, int sentencePosition, bool final ) const {
      if (covered_ || started_) {
        return 0.0;
      }
      if (final) {
        return penaltyWeight;
      }
      if (sentencePosition < (int)option_->GetStart()) {
        return 0.0;
      }
      size_t delay = sentencePosition - option_->GetStart();
      if (delay > windowSize) {
        return penaltyWeight;
      }
      return penaltyWeight * ((float)delay / (float)windowSize);
    }

  private:
    const XmlOption *option_;
    size_t position_;
    bool started_;
    bool covered_;
};


class Sentence {
  public:

    Sentence(const God &god, size_t vLineNum, const std::string& line);
    Sentence(const God &god, size_t vLineNum, const std::vector<std::string>& words);
		Sentence(God &god, size_t lineNum, const std::vector<size_t>& words);

    const Words& GetWords(size_t index = 0) const;
    size_t size(size_t index = 0) const;
    size_t GetLineNum() const;

    bool isXmlTag(const std::string& tag);
    std::string ParseXmlTagAttribute(const std::string& tag, const std::string& attributeName);
    std::string TrimXml(const std::string& str);
    std::vector<std::string> TokenizeXml(const std::string& str);
    std::string ProcessXml(const God &god, const std::string& input);
    bool HasXmlOptions() { return xmlOptions_.size() > 0; }
    const std::vector<XmlOption>& GetXmlOptions() { return xmlOptions_; }

  private:
    std::vector<Words> words_;
    size_t lineNum_;
    std::vector<XmlOption> xmlOptions_;

    Sentence(const Sentence &) = delete;
};

using SentencePtr = std::shared_ptr<Sentence>;

}

