def get_text_with_option_words(dp, include_option5 = False):
    if include_option5 == False:
        return f"(A) {dp['option_0']} (B) {dp['option_1']} (C) {dp['option_2']} (D) {dp['option_3']} (E) {dp['option_4']}"
    else:
        return f"(A) {dp['option_0']} (B) {dp['option_1']} (C) {dp['option_2']} (D) {dp['option_3']} (E) {dp['option_4']} (F) {dp['option_5']}"
    


def get_options_string_based_on_order(dp, order):
    option_letters = ["A", "B", "C", "D", "E", "F"]
    options = [f"option_{o}" for o in order]
    return_str = " ".join([f"({option_letter}) {dp[order_idx]}" for option_letter, order_idx in zip(option_letters[:len(order)], options)])
    return return_str