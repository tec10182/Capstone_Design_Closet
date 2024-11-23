import 'package:flutter/material.dart';
import 'package:closet/screens/login_screen.dart';
import 'package:closet/Components/onboarding_data.dart';
import 'package:shared_preferences/shared_preferences.dart';

class OnboardingPage extends StatefulWidget {
  const OnboardingPage({super.key});

  @override
  State<OnboardingPage> createState() => _OnboardingPageState();
}

class _OnboardingPageState extends State<OnboardingPage> {
  final controller = OnboardingData();
  final pageController = PageController();
  int currentIndex = 0;

  @override
  Widget build(BuildContext context) {
    bool isLastPage = currentIndex == controller.items.length - 1; // 마지막 페이지 여부

    return Scaffold(
      body: Column(
        children: [
          Expanded(child: buildPageView()),
          buildDots(),
          buildButton(isLastPage),
        ],
      ),
    );
  }

  // 페이지뷰 구성
  Widget buildPageView() {
    return PageView.builder(
      controller: pageController,
      onPageChanged: (index) {
        setState(() {
          currentIndex = index;
        });
      },
      itemCount: controller.items.length,
      itemBuilder: (context, index) {
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // 이미지
              Image.asset(controller.items[index].image),
              const SizedBox(height: 15),
              // 제목
              Text(
                controller.items[index].title,
                style: const TextStyle(
                  fontSize: 25,
                  color: Color(0xFF274a99),
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 10),
              // 설명
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 25),
                child: Text(
                  controller.items[index].description,
                  style: const TextStyle(color: Colors.grey, fontSize: 16),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  // 하단 점 표시
  Widget buildDots() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: List.generate(
        controller.items.length,
        (index) => AnimatedContainer(
          margin: const EdgeInsets.symmetric(horizontal: 2),
          height: 7,
          width: currentIndex == index ? 30 : 7,
          decoration: BoxDecoration(
            color:
                currentIndex == index ? const Color(0xFF274a99) : Colors.grey,
            borderRadius: BorderRadius.circular(50),
          ),
          duration: const Duration(milliseconds: 300),
        ),
      ),
    );
  }

  // 버튼 구성
  Widget buildButton(bool isLastPage) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 20),
      width: MediaQuery.of(context).size.width * 0.8,
      height: 55,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        color: const Color(0xFF274a99),
      ),
      child: TextButton(
        onPressed: () {
          if (isLastPage) {
            completeOnboarding(); // Onboarding 완료 처리
          } else {
            pageController.nextPage(
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeInOut,
            );
          }
        },
        child: Text(
          isLastPage ? "Get Started" : "Continue",
          style: const TextStyle(color: Colors.white, fontSize: 16),
        ),
      ),
    );
  }

  // Onboarding 완료 처리
  void completeOnboarding() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('onboardingCompleted', true);

    if (mounted) {
      // BuildContext가 여전히 유효한지 확인
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => const LoginScreen()),
      );
    }
  }
}
