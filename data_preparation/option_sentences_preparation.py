def get_text_with_option_words(dp, include_option5 = False):
    if include_option5 == False:
        return f"(A) {dp['option_0']} (B) {dp['option_1']} (C) {dp['option_2']} (D) {dp['option_3']} (E) {dp['option_4']}"
    else:
        return f"(A) {dp['option_0']} (B) {dp['option_1']} (C) {dp['option_2']} (D) {dp['option_3']} (E) {dp['option_4']} (F) {dp['option_5']}"