#include "sentence.h"
#include "god.h"
#include "utils.h"
#include "common/vocab.h"

namespace amunmt {

Sentence::Sentence(const God &god, size_t vLineNum, const std::string& line)
  : lineNum_(vLineNum)
{
  std::vector<std::string> tabs;
  Split(line, tabs, "\t");
  if (tabs.size() == 0) {
    tabs.push_back("");
  }
  if (god.Get<bool>("xml-input")) {
    tabs[0] = ProcessXml(god, tabs[0]);
  }

  size_t maxLength = god.Get<size_t>("max-length");
  size_t i = 0;
  for (auto& tab : tabs) {
    std::vector<std::string> lineTokens;
    Trim(tab);
    Split(tab, lineTokens, " ");

    if (maxLength && lineTokens.size() > maxLength) {
      lineTokens.resize(maxLength);
    }

    auto processed = god.Preprocess(i, lineTokens);
    words_.push_back(god.GetSourceVocab(i++)(processed));
  }
}

Sentence::Sentence(const God &god, size_t lineNum, const std::vector<std::string>& words)
  : lineNum_(lineNum) {
    auto processed = god.Preprocess(0, words);
    words_.push_back(god.GetSourceVocab(0)(processed));
}

Sentence::Sentence(God&, size_t lineNum, const std::vector<size_t>& words)
  : lineNum_(lineNum) {
    words_.push_back(words);
}


size_t Sentence::GetLineNum() const {
  return lineNum_;
}

const Words& Sentence::GetWords(size_t index) const {
  return words_[index];
}

size_t Sentence::size(size_t index) const {
  return words_[index].size();
}

//// code to handle XML tags... partly borrowed from Moses

std::vector<std::string> Sentence::TokenizeXml(const std::string& str) {
  std::vector<std::string> tokens; // vector of tokens to be returned
  std::string::size_type cpos = 0; // current position in string
  std::string::size_type lpos = 0; // left start of xml tag
  std::string::size_type rpos = 0; // right end of xml tag

  // walk thorugh the string (loop vver cpos)
  while (cpos != str.size()) {

    // find the next opening "<" of an xml tag
    lpos = str.find("<", cpos);
    if (lpos != std::string::npos) {
      // find the end of the xml tag
      rpos = str.find(">", lpos);
      // sanity check: there has to be closing ">"
      if (rpos == std::string::npos) {
        std::cerr << "XML ERROR (line " << lineNum_ << "): malformed XML: " << str << std::endl;
        return tokens;
      }
    }
    else { // no more tags found
      // add the rest as token
      tokens.push_back(str.substr(cpos));
      break;
    }

    // add stuff before xml tag as token, if there is any
    if (lpos - cpos > 0)
      tokens.push_back(str.substr(cpos, lpos - cpos));

    // add xml tag as token
    tokens.push_back(str.substr(lpos, rpos-lpos+1));
    cpos = rpos + 1;
  }
  return tokens;
}
/**
 * Check if the token is an XML tag, i.e. starts with "<"
 *
 * \param tag token to be checked
 * \param lbrackStr xml tag's left bracket string, typically "<"
 * \param rbrackStr xml tag's right bracket string, typically ">"
 */
bool Sentence::isXmlTag(const std::string& tag) {
  return tag[0] == '<' &&
         tag[tag.size()-1] == '>' &&
         ( tag[1] == '/'
           || (tag[1] >= 'a' && tag[1] <= 'z')
           || (tag[1] >= 'A' && tag[1] <= 'Z') );
}

/**
 * Remove "<" and ">" from XML tag
 */
std::string Sentence::TrimXml(const std::string& str) {
  // too short to be xml token -> do nothing
  if (str.size() < 2) return str;

  // strip first and last character
  if (isXmlTag(str)) {
    return str.substr(1, str.size()-2);
  }
  // not an xml token -> do nothing
  else {
    return str;
  }
}

/**
 * Get value for XML attribute, if it exists
 */
std::string Sentence::ParseXmlTagAttribute(const std::string& tag, const std::string& attributeName) {
  std::string tagOpen = attributeName + "=\"";
  size_t contentsStart = tag.find(tagOpen);
  if (contentsStart == std::string::npos) return "";
  contentsStart += tagOpen.size();
  size_t contentsEnd = tag.find_first_of('"',contentsStart+1);
  if (contentsEnd == std::string::npos) {
    std::cerr << "XML ERROR (line " << lineNum_ << "): Malformed XML attribute: "<< tag << std::endl;
    return "";
  }
  size_t possibleEnd;
  while (tag.at(contentsEnd-1) == '\\' && (possibleEnd = tag.find_first_of('"',contentsEnd+1)) != std::string::npos) {
    contentsEnd = possibleEnd;
  }
  return tag.substr(contentsStart,contentsEnd-contentsStart);
}
/**
 * Process a sentence with xml annotation
 * Xml tags may specifiy translation options
 * (PLANNED: and reordering constraints)
 */

std::string Sentence::ProcessXml(const God &god, const std::string& line) {
  // no xml tag? we're done.
  if (line.find("<") == std::string::npos) {
    return line;
  }

  // break up input into a vector of xml tags and text
  // example: (this), (<b>), (is a), (</b>), (test .)
  std::vector<std::string> xmlTokens = TokenizeXml(line);

  // we need to store opened tags, until they are closed
  // tags are stored as tripled (tagname, startpos, contents)
  typedef std::pair< std::string, std::pair< size_t, std::string > > OpenedTag;
  std::vector< OpenedTag > tagStack; // stack that contains active opened tags

  std::string cleanLine; // return string (text without xml)
  size_t wordPos = 0; // position in sentence (in terms of number of words)

  // loop through the tokens
  for (size_t xmlTokenPos = 0 ; xmlTokenPos < xmlTokens.size() ; xmlTokenPos++) {

    // not a xml tag, but regular text (may contain many words)
    if (!isXmlTag(xmlTokens[xmlTokenPos])) {
      // add a space at boundary, if necessary
      if (cleanLine.size()>0 &&
          cleanLine[cleanLine.size() - 1] != ' ' &&
          xmlTokens[xmlTokenPos][0] != ' ') {
        cleanLine += " ";
      }
      cleanLine += xmlTokens[xmlTokenPos]; // add to output
      std::vector< std::string > outputWords;
      Split(cleanLine, outputWords, " ");
      wordPos = outputWords.size(); // count all the words
    }

    // process xml tag
    else {
      // first: get essential information about tag
      std::string tag = TrimXml(xmlTokens[xmlTokenPos]);
      Trim(tag);

      if (tag.size() == 0) {
        return line;
      }

      // check if unary (e.g., "<wall/>")
      bool isUnary = ( tag[tag.size() - 1] == '/' );

      // check if opening tag (e.g. "<a>", not "</a>")
      bool isClosed = ( tag[0] == '/' );
      bool isOpen = !isClosed;

      if (isClosed && isUnary) {
        std::cerr << "XML ERROR (line " << lineNum_ << "): can't have both closed and unary tag <" << tag << ">: " << line << std::endl;
        return line;
      }

      if (isClosed)
        tag = tag.substr(1); // remove "/" at the beginning
      if (isUnary)
        tag = tag.substr(0,tag.size()-1); // remove "/" at the end 

      // find the tag name and contents
      std::string::size_type endOfName = tag.find_first_of(' ');
      std::string tagName = tag;
      std::string tagContent = "";
      if (endOfName != std::string::npos) {
        tagName = tag.substr(0,endOfName);
        tagContent = tag.substr(endOfName+1);
      }

      // process new tag
      if (isOpen || isUnary) {
        OpenedTag openedTag = std::make_pair( tagName, std::make_pair( wordPos, tagContent ) );
        tagStack.push_back( openedTag );
      }

      // process completed tag
      if (isClosed || isUnary) {

        // pop last opened tag from stack;
        if (tagStack.size() == 0) {
          std::cerr << "XML ERROR (line " << lineNum_ << "): tag " << tagName << " closed, but not opened" << ":" << line << std::endl;
          return line;
        }
        OpenedTag openedTag = tagStack.back();
        tagStack.pop_back();

        // tag names have to match
        if (openedTag.first != tagName) {
          std::cerr << "XML ERROR (line " << lineNum_ << "): tag " << openedTag.first << " closed by tag " << tagName << ": " << line << std::endl;
          return line;
        }

        // assemble remaining information about tag
        size_t startPos = openedTag.second.first;
        std::string& tagContent = openedTag.second.second;
        size_t endPos = wordPos;

        // span attribute overwrites position
       std::string span = ParseXmlTagAttribute(tagContent,"span");
        if (! span.empty()) {
          std::vector<std::string> ij;
          Split(span, ij, "-");
          if (ij.size() != 1 && ij.size() != 2) {
            std::cerr << "XML ERROR (line " << lineNum_ << "): span attribute must be of the form \"i-j\" or \"i\": " << line << std::endl;
            return line;
          }
          startPos = atoi(ij[0].c_str());
          if (ij.size() == 1) endPos = startPos + 1;
          else endPos = atoi(ij[1].c_str()) + 1;
        }

        // TODO: special tag: wall
        // TODO: special tag: zone

        // default: opening tag that specifies translation options
        else {
          if (startPos > endPos) {
            std::cerr << "XML ERROR (line " << lineNum_ << "): tag " << tagName << " startPos > endPos: " << line << std::endl;
            return line;
          }
          else if (startPos == endPos) {
            std::cerr << "XML ERROR (line " << lineNum_ << "): tag " << tagName << " span: " << line << std::endl;
            continue;
          }

          // specified translations -> vector of phrases
          std::string translation = ParseXmlTagAttribute(tagContent,"translation");
          if (translation.empty()) {
            translation = ParseXmlTagAttribute(tagContent,"english");
          }
          if (translation.empty()) {
            continue;
          }
          std::vector< std::string > translationWords;
          Trim(translation);
          Split(translation, translationWords, " ");
          std::cerr << "new option (" << startPos << "," << endPos << ") " << translation << ", size " << translationWords.size() << "\n";
          xmlOptions_.push_back( XmlOption(startPos, endPos, god.GetTargetVocab()(translationWords, false)) );
        }
      }
    }
  }
  return cleanLine;
}


}

