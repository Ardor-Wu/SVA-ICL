void LinkChangeSerializerMarkupAccumulator::non_vulnerable_func(StringBuilder& result, Element* element, const Attribute& attribute, Namespaces* namespaces)
{
    if (m_replaceLinks && element->isURLAttribute(attribute) && !element->isJavaScriptURLAttribute(attribute)) {
        String completeURL = m_document->completeURL(attribute.value());
        if (m_replaceLinks->contains(completeURL)) {
            result.append(' ');
            result.append(attribute.name().toString());
            result.appendLiteral("=\"");
            if (!m_directoryName.isEmpty()) {
                result.appendLiteral("./");
                result.append(m_directoryName);
                result.append('/');
            }
            result.append(m_replaceLinks->get(completeURL));
            result.appendLiteral("\"");
            return;
        }
    }
    MarkupAccumulator::non_vulnerable_func(result, element, attribute, namespaces);
}
