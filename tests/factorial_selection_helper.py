"""Drive the interaction checkbox grid through its visible controls."""
from utils.factorial_interactions_ui import ALL_INTERACTIONS, NO_INTERACTION

class InteractionSelection:
    def __init__(self, app):
        self.app = app

    @property
    def value(self):
        return self.app.session_state['fac_interaction_choices']

    def set_value(self, pairs):
        app = self.app
        if ALL_INTERACTIONS in pairs:
            app.button(key='fac_action_interactions_all').click().run(timeout=60)
        else:
            app.button(key='fac_action_interactions_clear').click().run(timeout=60)
            for pair in pairs:
                if pair == NO_INTERACTION:
                    continue
                label = ' × '.join(pair)
                next(c for c in app.checkbox if c.label == label).check().run(timeout=60)
        return app

def interaction_selection(app):
    return InteractionSelection(app)
