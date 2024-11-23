import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:closet/screens/login_screen.dart';
import 'package:closet/screens/onboarding.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized(); // 초기화 보장
  await Future.delayed(const Duration(seconds: 3)); // 3초 지연
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  Future<bool> hasSeenOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool('onboardingCompleted') ?? false;
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<bool>(
      future: hasSeenOnboarding(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const CircularProgressIndicator(); // 로딩 상태 표시
        } else {
          final seenOnboarding = snapshot.data ?? false;
          return MaterialApp(
            debugShowCheckedModeBanner: false,
            title: 'CLOSET App',
            theme: ThemeData(
              colorScheme:
                  ColorScheme.fromSeed(seedColor: const Color(0xFF274a99)),
              useMaterial3: true,
            ),
            home: seenOnboarding ? const LoginScreen() : const OnboardingPage(),
          );
        }
      },
    );
  }
}
