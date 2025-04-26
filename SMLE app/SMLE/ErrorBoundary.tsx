import React, { Component, ReactNode } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';

type Props = {
  children: ReactNode;
};

type State = {
  hasError: boolean;
  error: Error | null;
};

class ErrorBoundary extends Component<Props, State> {
  state: State = {
    hasError: false,
    error: null,
  };

  static getDerivedStateFromError(error: Error): State {
    // Aktualizuj stan, aby kolejny render pokazał UI zastępcze
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    // Możesz zalogować błąd do zewnętrznego serwisu, np. Sentry
    console.error('Wystąpił błąd w ErrorBoundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.container}>
          <Text style={styles.errorText}>Coś poszło nie tak.</Text>
          <Text style={styles.errorDetails}>
            {this.state.error?.message || 'Nieznany błąd'}
          </Text>
          <TouchableOpacity
            style={styles.button}
            onPress={() => {
              this.setState({ hasError: false, error: null });
            }}
          >
            <Text style={styles.buttonText}>Spróbuj ponownie</Text>
          </TouchableOpacity>
        </View>
      );
    }

    return this.props.children;
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#121212',
    padding: 20,
  },
  errorText: {
    color: '#FF4444',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  errorDetails: {
    color: '#FFF',
    fontSize: 16,
    marginBottom: 20,
    textAlign: 'center',
  },
  button: {
    backgroundColor: '#00A1D6',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  buttonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default ErrorBoundary;